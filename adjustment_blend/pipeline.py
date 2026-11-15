"""
Pipeline orchestration for Adjustment Blend.

Ties the pieces together in four stages:

1. :func:`build_context`        — validate the scene state, find the layer's curves.
2. :func:`collect_attribute_data` — read composite motion via :class:`LayerStack`.
3. :func:`calculate_adjustment_values` — distribute the adjustment (pure core).
4. :func:`apply_keyframes`      — write the result back to Maya, as one undo step.

Every curve is blended between its own keys. Two controls keyed on different
frames, or two attributes of one control, do not cut each other into segments.

:func:`run` is the public entry point. Call it with no arguments to operate on
the current selection and the Anim Layer editor's state, or pass
``adjustment_layer`` / ``layers_below`` / ``objects`` to drive it explicitly
without touching the UI.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from maya import cmds

from . import core
from . import maya_keys
from . import maya_scene
from . import vector_core
from .maya_layers import LayerStack
from .maya_scene import LayerCurve

log = logging.getLogger(__name__)


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class AdjustmentContext:
    """The "what to process" gathered up front for one adjustment-blend run."""
    adjustment_layer: str
    layers_to_process: List[str]
    objects: List[str]
    curves: List[LayerCurve]           # the adjustment layer's curves for the objects
    calculation_range: List[float]     # every frame sampled: first key of any curve to the last
    stack: LayerStack                  # samples the motion below, shared by the whole run


@dataclass
class AttributeData:
    """Everything needed to blend a single ``object.attribute``.

    The lists hold one sample per entry of ``frames``, which runs from this
    curve's first key to its last.
    """
    obj: str
    attr: str
    adjustment_curve: str
    span: slice                        # where ``frames`` sits in the run's calculation range
    frames: List[float]
    key_ranges: List[Tuple[float, float]]   # (start, end) between this curve's neighboring keys
    adjustment_values: List[float]     # the adjustment curve's own value
    composite_velocity: List[float]    # speed of the motion below

    @property
    def full_attr(self) -> str:
        return f"{self.obj}.{self.attr}"

    def has_valid_velocity(self) -> bool:
        """True if the composite velocity carries any motion to blend against."""
        if not self.composite_velocity:
            return False
        return not core.is_still(self.composite_velocity)


# =============================================================================
# Stage 1: Build context
# =============================================================================

def build_context(
    adjustment_layer: str,
    layers_below: List[str],
    objects: Optional[List[str]] = None,
) -> Optional[AdjustmentContext]:
    """Validate scene state and find the adjustment curves to work on.

    Args:
        adjustment_layer: The layer whose sparse keys are being densified.
        layers_below: The ordered layers to composite motion from.
        objects: Objects to process. ``None`` falls back to the current
            selection, then to the adjustment layer's members.

    Returns:
        A populated :class:`AdjustmentContext`, or ``None`` if validation fails
        (a ``cmds.warning`` explains why in every failure case).
    """
    if not layers_below:
        cmds.warning("No animation layers below adjustment layer to process. Aborting!")
        return None

    adjustment_layer_members = cmds.animLayer(adjustment_layer, q=True, attribute=True) or []
    if not adjustment_layer_members:
        cmds.warning(f"Adjustment layer {adjustment_layer} has no members. Aborting!")
        return None

    members = list(dict.fromkeys(x.split(".")[0] for x in adjustment_layer_members))

    # Resolve objects: explicit arg -> selection -> layer members.
    if objects is None:
        objects = cmds.ls(sl=1) or []

    if not objects:
        cmds.warning(f"No object selected. Fetching members of {adjustment_layer} instead.")
        objects = members
    elif not bool(set(members) & set(objects)):
        cmds.warning("No selected objects exist in the selected adjustment layer!")
        return None

    if not objects:
        cmds.warning(f"{adjustment_layer} contains no controls. Aborting!")
        return None

    # Drop layers that don't touch any of our objects.
    layers_below = maya_scene.filter_layers_by_objects(layers_below, objects)

    curves = maya_scene.layer_curves(adjustment_layer, objects)
    adjustment_keys = sorted({key for curve in curves for key in curve.keys})
    if not adjustment_keys:
        cmds.warning(f"Could not find any adjustment keys on {adjustment_layer}")
        return None
    if len(adjustment_keys) < 2:
        cmds.warning(f"Need at least 2 adjustment keys on {adjustment_layer}")
        return None

    calculation_range = core.get_float_range(adjustment_keys)

    log.debug("Adjustment layer: %s", adjustment_layer)
    log.debug("Layers below: %s", layers_below)
    log.debug("Objects: %s", objects)
    log.debug("Frames: %s to %s", calculation_range[0], calculation_range[-1])

    return AdjustmentContext(
        adjustment_layer=adjustment_layer,
        layers_to_process=layers_below,
        objects=objects,
        curves=curves,
        calculation_range=calculation_range,
        stack=LayerStack(adjustment_layer, calculation_range),
    )


# =============================================================================
# Stage 2: Collect attribute data (using LayerStack)
# =============================================================================

def collect_attribute_data(context: AdjustmentContext) -> List[AttributeData]:
    """Read the adjustment curve and composite motion for every eligible attr."""
    attribute_data_list: List[AttributeData] = []
    skipped: Dict[str, list] = {
        "fewer_than_two_keys": [],
        "flat_adjustment": [],
        "no_composite_velocity": [],
    }

    grid = context.calculation_range
    position = {frame: index for index, frame in enumerate(grid)}

    for curve in context.curves:
        attribute = f"{curve.obj}.{curve.attr}"
        if len(curve.keys) < 2:
            skipped["fewer_than_two_keys"].append(attribute)
            continue

        # This curve's own stretch of the run: its first key to its last.
        span = slice(position[curve.keys[0]], position[curve.keys[-1]] + 1)

        adjustment_values = context.stack.curve_values(curve.curve)[span]
        if core.is_equal(adjustment_values):
            skipped["flat_adjustment"].append(attribute)
            continue

        # Composite motion of every layer below, via LayerStack.
        composite_values = context.stack.base_values(curve.obj, curve.attr)
        if composite_values:
            composite_velocity = core.get_velocity_graph(composite_values[span])
        else:
            composite_velocity = [0.0] * len(adjustment_values)

        attr_data = AttributeData(
            obj=curve.obj,
            attr=curve.attr,
            adjustment_curve=curve.curve,
            span=span,
            frames=grid[span],
            key_ranges=list(zip(curve.keys, curve.keys[1:])),
            adjustment_values=adjustment_values,
            composite_velocity=composite_velocity,
        )

        if not attr_data.has_valid_velocity():
            # Keep it: smart fallback may find motion on a sibling attr.
            skipped["no_composite_velocity"].append(attribute)

        attribute_data_list.append(attr_data)

    for reason, attrs in skipped.items():
        if attrs:
            log.debug("Skipped (%s): %s", reason, attrs)

    return attribute_data_list


# =============================================================================
# Stage 3: Calculate adjustment values (delegates to the pure core)
# =============================================================================

def calculate_adjustment_values(attr_data: AttributeData) -> List[float]:
    """Distribute the adjustment across frames by motion (see :mod:`core`)."""
    return core.distribute_adjustment(
        attr_data.composite_velocity,
        attr_data.adjustment_values,
        attr_data.key_ranges,
        attr_data.frames,
    )


# =============================================================================
# Stage 4: Apply keyframes
# =============================================================================

def apply_keyframes(
    results: List[Tuple[AttributeData, List[float]]],
    context: AdjustmentContext,
) -> bool:
    """Write the densified values back onto the adjustment curves.

    Each curve is keyed on the whole frames between its own first and last
    key. Every curve is written in one undo step (see :mod:`maya_keys`). If any
    of them cannot be written, none is.
    """
    # A key between two frames puts its time in the samples. It keeps its own
    # value, and no new key is set there.
    subframes = any(not frame.is_integer() for frame in context.calculation_range)

    writes = []
    for attr_data, new_values in results:
        frames = attr_data.frames
        if subframes:
            whole = [i for i, frame in enumerate(frames) if frame.is_integer()]
            frames = [frames[i] for i in whole]
            new_values = [new_values[i] for i in whole]
        if frames:
            writes.append((attr_data.adjustment_curve, frames, new_values))

    try:
        maya_keys.write_keys(writes)
        return True
    except Exception as e:  # noqa: BLE001
        log.error("Failed to apply keyframes, nothing was changed: %s", e)
        return False


# =============================================================================
# Smart fallback: borrow motion from a sibling axis or channel
# =============================================================================

def apply_smart_fallbacks(
    attribute_data_list: List[AttributeData],
    context: AdjustmentContext,
) -> List[AttributeData]:
    """Fill flat composites with motion borrowed from related attributes.

    When an attribute's own layers-below motion is flat, we look for usable
    motion on:

    1. the other axes of the same channel (``rotateY`` -> ``rotateX``/``rotateZ``), then
    2. the other channels entirely (``rotate`` -> ``translate`` -> ``scale``).

    The motion is always the sibling's own, read from the run's shared
    :class:`LayerStack` over this curve's frames. So we can borrow from a
    sibling attribute even if it isn't keyed on the adjustment layer, and the
    order the attributes are processed in does not matter.
    """
    stack = context.stack

    for attr_data in attribute_data_list:
        if attr_data.has_valid_velocity():
            continue

        # Same channel first, then other channels.
        fallback_velocity, fallback_source = _find_axis_fallback(attr_data, stack)
        if not fallback_velocity:
            fallback_velocity, fallback_source = _find_channel_fallback(attr_data, stack)

        if fallback_velocity:
            attr_data.composite_velocity = fallback_velocity
            log.debug("Substituting %s velocity for %s", fallback_source, attr_data.attr)

    return attribute_data_list


def _motion_below(
    attr_data: AttributeData,
    attr: str,
    stack: LayerStack,
) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """Return ``(values, velocity)`` of another attribute of the same object.

    Sampled over ``attr_data``'s own frames. ``(None, None)`` when that
    attribute does not move there.
    """
    values = stack.base_values(attr_data.obj, attr)
    if not values:
        return None, None
    values = values[attr_data.span]
    velocity = core.get_velocity_graph(values)
    if core.is_still(velocity):
        return None, None
    return values, velocity


def _find_axis_fallback(
    attr_data: AttributeData,
    stack: LayerStack,
) -> Tuple[Optional[List[float]], Optional[str]]:
    """Borrow motion from another axis of the same channel.

    Picks the axis whose net displacement (end - start) is closest to the
    adjustment curve's net displacement. Artistically: a stride that travels
    0 -> 50 is a better match for a 0 -> 45 horizontal adjustment than an
    up/down bob that ends where it started.
    """
    adj_values = attr_data.adjustment_values
    adjustment_delta = abs(adj_values[-1] - adj_values[0])

    candidates = []
    for other_attr in core.get_other_axis(attr_data.attr):
        values, velocity = _motion_below(attr_data, other_attr, stack)
        if velocity:
            candidates.append((other_attr, velocity, abs(values[-1] - values[0])))

    if not candidates:
        return None, None
    if len(candidates) == 1:
        return candidates[0][1], candidates[0][0]

    best = min(candidates, key=lambda c: abs(c[2] - adjustment_delta))

    candidate_info = ", ".join(
        f"{c[0]}={c[2]:.2f} (diff={abs(c[2] - adjustment_delta):.2f})" for c in candidates
    )
    log.debug(
        "Axis fallback for %s: adjustment_delta=%.2f, candidates: [%s], chose %s",
        attr_data.attr, adjustment_delta, candidate_info, best[0],
    )
    return best[1], best[0]


def _find_channel_fallback(
    attr_data: AttributeData,
    stack: LayerStack,
) -> Tuple[Optional[List[float]], Optional[str]]:
    """Borrow motion from another channel, choosing the hottest one available."""
    best_velocity = None
    best_source = None
    best_max_vel = 0.0

    for channel in core.get_other_channel(attr_data.attr):
        for axis in ["X", "Y", "Z"]:
            other_attr = f"{channel}{axis}"
            _, velocity = _motion_below(attr_data, other_attr, stack)
            if velocity:
                max_vel = max(velocity)
                if max_vel > best_max_vel:
                    best_max_vel = max_vel
                    best_velocity = velocity
                    best_source = other_attr

    return best_velocity, best_source


# =============================================================================
# Public entry point
# =============================================================================

def run(
    adjustment_layer: Optional[str] = None,
    layers_below: Optional[List[str]] = None,
    objects: Optional[List[str]] = None,
    *,
    signal: str = "scalar",
    smart: bool = False,
    apply: bool = True,
) -> bool:
    """Run an adjustment blend.

    Called with no arguments, it discovers the adjustment layer and the stack
    below it from the Anim Layer editor and operates on the current selection —
    the everyday interactive workflow::

        import adjustment_blend
        adjustment_blend.run(smart=True)

    Supply the layer context explicitly to run without the UI (headless,
    pipeline, tests)::

        adjustment_blend.run(
            adjustment_layer="AnimLayer1",
            layers_below=["BaseAnimation"],
            objects=["pCube1"],
        )

    Args:
        adjustment_layer: Target layer to densify. Discovered from the UI if
            ``None``.
        layers_below: Ordered layers to composite motion from. Discovered from
            the UI if ``None``.
        objects: Objects to process. Defaults to the selection, then to the
            adjustment layer's members.
        signal: Which motion signal drives the distribution.

            * ``"scalar"`` (default) — per-attribute velocity. ``smart`` borrows
              motion from a sibling axis, then another channel, when an
              attribute's own composite is flat.
            * ``"vector"`` (experimental) — one shared speed per channel group
              (translation speed / angular speed / scale speed). A flat axis
              rides its group's motion automatically, so ``smart`` only governs
              the last-resort *cross-channel* borrow (e.g. a fully static
              rotation group riding translation).
        smart: Enable fallbacks for attributes with no motion of their own
            (see ``signal``).
        apply: If ``True``, write keyframes. If ``False``, compute only (dry run).

    Returns:
        ``True`` if any attribute was processed, ``False`` otherwise.
    """
    # Stage 0: resolve the layer context (explicit args win; else discover).
    if adjustment_layer is None or layers_below is None:
        discovered = maya_scene.discover_layers()
        if not discovered:
            cmds.warning("No animation layers to process. Aborting!")
            return False
        discovered_layer, discovered_below = discovered
        if adjustment_layer is None:
            adjustment_layer = discovered_layer
        if layers_below is None:
            layers_below = discovered_below

    # Stage 1: context
    context = build_context(adjustment_layer, layers_below, objects)
    if not context:
        return False

    if signal == "vector":
        return _run_vector(context, smart=smart, apply=apply)
    if signal == "scalar":
        return _run_scalar(context, smart=smart, apply=apply)
    raise ValueError(f"Unknown signal {signal!r}; expected 'scalar' or 'vector'.")


def _finish(results, context, apply, label):
    """Apply (or, in a dry run, just report) every attribute's new values."""
    processed = [attr_data.full_attr for attr_data, _ in results]
    if not apply:
        cmds.warning(
            f"apply=False: dry run, no keyframes written. Would adjust ({label}): {processed}"
        )
        return bool(processed)

    if results and not apply_keyframes(results, context):
        return False
    log.info("Adjusted attributes (%s): %s", label, processed)
    return bool(processed)


def _run_scalar(context: AdjustmentContext, *, smart: bool, apply: bool) -> bool:
    """Original per-attribute velocity distribution."""
    attribute_data_list = collect_attribute_data(context)
    if not attribute_data_list:
        cmds.warning("No attributes to process!")
        return False

    if smart:
        attribute_data_list = apply_smart_fallbacks(attribute_data_list, context)

    valid_attributes = [ad for ad in attribute_data_list if ad.has_valid_velocity()]
    if not valid_attributes:
        cmds.warning("No attributes have valid velocity data to process!")
        return False

    results = [
        (attr_data, calculate_adjustment_values(attr_data))
        for attr_data in valid_attributes
    ]
    return _finish(results, context, apply, "scalar")


def _run_vector(context: AdjustmentContext, *, smart: bool, apply: bool) -> bool:
    """Experimental per-channel-group speed distribution (see :mod:`vector_core`)."""
    attribute_data_list = collect_attribute_data(context)
    if not attribute_data_list:
        cmds.warning("No attributes to process!")
        return False

    # Which channel groups does each object have adjustments on?
    obj_groups: Dict[str, set] = defaultdict(set)
    for ad in attribute_data_list:
        group = vector_core.channel_of(ad.attr)
        if group:
            obj_groups[ad.obj].add(group)

    # Compute the shared speed signal per (object, group). When smart is on we
    # compute every group so a static group can borrow from a moving one.
    obj_speeds: Dict[str, Dict[str, List[float]]] = {}
    for obj, groups in obj_groups.items():
        needed = set(core.CHANNELS) if smart else set(groups)
        obj_speeds[obj] = _group_speeds(obj, needed, context.stack)

    results: List[Tuple[AttributeData, List[float]]] = []
    left_sparse: List[str] = []
    for ad in attribute_data_list:
        if ad.obj not in obj_speeds:
            continue

        # The group speeds over this curve's own frames. The first sample is
        # the motion that arrived at its first key, which is not its to follow.
        speeds = {
            name: [0.0] + speed[ad.span][1:]
            for name, speed in obj_speeds[ad.obj].items()
        }

        group = vector_core.channel_of(ad.attr)
        speed = speeds.get(group)
        source = group

        # Nothing on this group's own motion: optionally borrow another channel.
        if speed is None or core.is_still(speed):
            borrow = vector_core.hottest_group(speeds, exclude=group) if smart else None
            if borrow:
                source, speed = borrow

        if speed is None or core.is_still(speed):
            # No motion to ride anywhere — leave the attr sparse (Maya's own
            # linear blend already covers a motionless offset).
            left_sparse.append(ad.full_attr)
            continue

        if source != group:
            log.debug("Vector: %s riding %s-group speed", ad.attr, source)

        new_values = vector_core.distribute_by_speed(
            speed, ad.adjustment_values, ad.key_ranges, ad.frames,
        )
        results.append((ad, new_values))

    if left_sparse:
        log.debug("Vector: no motion to ride, left sparse: %s", left_sparse)

    return _finish(results, context, apply, "vector")


def _group_speeds(
    obj: str,
    groups: set,
    stack: LayerStack,
) -> Dict[str, List[float]]:
    """Sample the below-motion for each axis of each group and reduce to speed."""
    speeds: Dict[str, List[float]] = {}
    flat = [0.0] * len(stack.times)

    for group in groups:
        axes: Dict[str, List[float]] = {}
        for axis in ("X", "Y", "Z"):
            axes[axis] = stack.base_values(obj, group + axis) or flat

        order = "xyz"
        if group == "rotate":
            try:
                order = cmds.getAttr(f"{obj}.rotateOrder")
            except Exception:  # noqa: BLE001 - fall back to a sane default order
                order = "xyz"

        speeds[group] = vector_core.channel_group_speed(group, axes, order)

    return speeds
