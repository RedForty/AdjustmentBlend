"""
Pipeline orchestration for Adjustment Blend.

Ties the pieces together in four stages:

1. :func:`build_context`        — validate the scene state, collect adjustment keys.
2. :func:`collect_attribute_data` — read composite motion via :class:`LayerStack`.
3. :func:`calculate_adjustment_values` — distribute the adjustment (pure core).
4. :func:`apply_keyframes`      — write the result back to Maya.

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
from . import maya_scene
from . import vector_core
from .maya_layers import LayerStack

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
    adjustment_keys: List[float]
    adjustment_key_ranges: List[Tuple[float, float]]
    calculation_range: List[float]


@dataclass
class AttributeData:
    """Everything needed to blend a single ``object.attribute``."""
    obj: str
    attr: str
    adjustment_curve: str
    adjustment_values: List[float]     # adjustment value at each calculation frame
    composite_velocity: List[float]    # motion of the layers below, per frame
    velocity_source: str = ""          # which attr the velocity came from (debugging)

    @property
    def full_attr(self) -> str:
        return f"{self.obj}.{self.attr}"

    def has_valid_velocity(self) -> bool:
        """True if the composite velocity carries any motion to blend against."""
        if not self.composite_velocity:
            return False
        return not core.is_equal(self.composite_velocity)


# =============================================================================
# Stage 1: Build context
# =============================================================================

def build_context(
    adjustment_layer: str,
    layers_below: List[str],
    objects: Optional[List[str]] = None,
) -> Optional[AdjustmentContext]:
    """Validate scene state and gather the adjustment keys to work between.

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

    members = list(set(x.split(".")[0] for x in adjustment_layer_members))

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

    adjustment_keys = maya_scene.collect_adjustment_keys(
        objects, adjustment_layer, adjustment_layer_members
    )
    if not adjustment_keys:
        cmds.warning(f"Could not find any adjustment keys on {adjustment_layer}")
        return None
    if len(adjustment_keys) < 2:
        cmds.warning(f"Need at least 2 adjustment keys on {adjustment_layer}")
        return None

    adjustment_keys_sorted = sorted(adjustment_keys)
    adjustment_key_ranges = [
        (adjustment_keys_sorted[i], adjustment_keys_sorted[i + 1])
        for i in range(len(adjustment_keys_sorted) - 1)
    ]
    calculation_range = core.get_float_range(list(adjustment_keys))

    log.debug("Adjustment layer: %s", adjustment_layer)
    log.debug("Layers below: %s", layers_below)
    log.debug("Objects: %s", objects)
    log.debug("Key ranges: %s", adjustment_key_ranges)

    return AdjustmentContext(
        adjustment_layer=adjustment_layer,
        layers_to_process=layers_below,
        objects=objects,
        adjustment_keys=adjustment_keys_sorted,
        adjustment_key_ranges=adjustment_key_ranges,
        calculation_range=calculation_range,
    )


# =============================================================================
# Stage 2: Collect attribute data (using LayerStack)
# =============================================================================

def collect_attribute_data(context: AdjustmentContext) -> List[AttributeData]:
    """Read the adjustment curve and composite motion for every eligible attr."""
    attribute_data_list: List[AttributeData] = []
    skipped: Dict[str, list] = {
        "no_adjustment_curve": [],
        "flat_adjustment": [],
        "no_composite_velocity": [],
    }

    adjustment_layer_members = set(
        cmds.animLayer(context.adjustment_layer, q=True, attribute=True) or []
    )

    for obj in context.objects:
        # LayerStack knows how to traverse blend nodes for composite values.
        try:
            stack = LayerStack.build(obj, context.adjustment_layer)
        except Exception as e:  # noqa: BLE001 - surface, don't abort the whole run
            log.warning("Could not build LayerStack for %s: %s", obj, e)
            continue

        for attribute in maya_scene.get_animated_attributes(obj):
            _, attr = attribute.split(".")

            if attr not in core.ATTRIBUTES:
                continue
            if attribute not in adjustment_layer_members:
                continue

            curve = cmds.animLayer(context.adjustment_layer, q=True, findCurveForPlug=attribute)
            if not curve:
                skipped["no_adjustment_curve"].append(attribute)
                continue
            curve = curve[0] if isinstance(curve, list) else curve

            adjustment_values = []
            for t in context.calculation_range:
                value = cmds.keyframe(curve, q=True, valueChange=True, eval=True, time=(t,))
                adjustment_values.append(value[0] if value else 0.0)

            if core.is_equal(adjustment_values):
                skipped["flat_adjustment"].append(attribute)
                continue

            # Composite motion of every layer below, via LayerStack.
            composite_values = stack.get_base_values(attr, context.calculation_range)
            composite_velocity = core.get_velocity_graph(composite_values)

            attr_data = AttributeData(
                obj=obj,
                attr=attr,
                adjustment_curve=curve,
                adjustment_values=adjustment_values,
                composite_velocity=composite_velocity,
                velocity_source=attr,
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

def calculate_adjustment_values(
    attr_data: AttributeData,
    context: AdjustmentContext,
) -> List[float]:
    """Distribute the adjustment across frames by motion (see :mod:`core`)."""
    return core.distribute_adjustment(
        attr_data.composite_velocity,
        attr_data.adjustment_values,
        context.adjustment_key_ranges,
        context.calculation_range,
    )


# =============================================================================
# Stage 4: Apply keyframes
# =============================================================================

def apply_keyframes(
    attr_data: AttributeData,
    new_values: List[float],
    context: AdjustmentContext,
) -> bool:
    """Write the densified values back onto the adjustment curve."""
    adjustment_range = range(
        int(context.adjustment_key_ranges[0][0]),
        int(context.adjustment_key_ranges[-1][1]) + 1,
    )

    try:
        for i, frame in enumerate(adjustment_range):
            if i >= len(new_values):
                break
            cmds.setKeyframe(
                attr_data.adjustment_curve,
                animLayer=context.adjustment_layer,
                time=(frame,),
                value=new_values[i],
            )
        return True
    except Exception as e:  # noqa: BLE001
        log.error("Failed to apply keyframes for %s: %s", attr_data.full_attr, e)
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

    The :class:`LayerStack` is queried directly, so we can borrow from a sibling
    attribute even if it isn't keyed on the adjustment layer.
    """
    lookup: Dict[str, AttributeData] = {
        f"{ad.obj}.{ad.attr}": ad for ad in attribute_data_list
    }
    layer_stacks: Dict[str, LayerStack] = {}

    for attr_data in attribute_data_list:
        if attr_data.has_valid_velocity():
            continue

        if attr_data.obj not in layer_stacks:
            try:
                layer_stacks[attr_data.obj] = LayerStack.build(
                    attr_data.obj, context.adjustment_layer
                )
            except Exception as e:  # noqa: BLE001
                log.warning("Could not build LayerStack for %s: %s", attr_data.obj, e)
                continue

        stack = layer_stacks[attr_data.obj]

        # Same channel first, then other channels.
        fallback_velocity, fallback_source = _find_axis_fallback(
            attr_data, lookup, stack, context.calculation_range
        )
        if not fallback_velocity:
            fallback_velocity, fallback_source = _find_channel_fallback(
                attr_data, lookup, stack, context.calculation_range
            )

        if fallback_velocity and not core.is_equal(fallback_velocity):
            attr_data.composite_velocity = fallback_velocity
            attr_data.velocity_source = fallback_source
            log.debug("Substituting %s velocity for %s", fallback_source, attr_data.attr)

    return attribute_data_list


def _find_axis_fallback(
    attr_data: AttributeData,
    lookup: Dict[str, AttributeData],
    stack: LayerStack,
    calculation_range: List[float],
) -> Tuple[Optional[List[float]], Optional[str]]:
    """Borrow motion from another axis of the same channel.

    Picks the axis whose net displacement (end - start) is closest to the
    adjustment curve's net displacement. Artistically: a stride that travels
    0 -> 50 is a better match for a 0 -> 45 horizontal adjustment than an
    up/down bob that ends where it started.
    """
    other_axes = core.get_other_axis(attr_data.attr)

    adj_values = attr_data.adjustment_values
    adjustment_delta = abs(adj_values[-1] - adj_values[0])

    candidates = []
    for other_attr in other_axes:
        key = f"{attr_data.obj}.{other_attr}"
        velocity = None
        composite_values = None

        if key in lookup and lookup[key].has_valid_velocity():
            velocity = lookup[key].composite_velocity[:]
            if stack.ensure_contribution(other_attr):
                composite_values = stack.get_base_values(other_attr, calculation_range)
        elif stack.ensure_contribution(other_attr):
            composite_values = stack.get_base_values(other_attr, calculation_range)
            velocity = core.get_velocity_graph(composite_values)

        if velocity and not core.is_equal(velocity) and composite_values:
            candidate_delta = abs(composite_values[-1] - composite_values[0])
            candidates.append((other_attr, velocity, candidate_delta))

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
    lookup: Dict[str, AttributeData],
    stack: LayerStack,
    calculation_range: List[float],
) -> Tuple[Optional[List[float]], Optional[str]]:
    """Borrow motion from another channel, choosing the hottest one available."""
    best_velocity = None
    best_source = None
    best_max_vel = 0.0

    for channel in core.get_other_channel(attr_data.attr):
        for axis in ["X", "Y", "Z"]:
            other_attr = f"{channel}{axis}"
            key = f"{attr_data.obj}.{other_attr}"

            if key in lookup and lookup[key].has_valid_velocity():
                velocity = lookup[key].composite_velocity
            elif stack.ensure_contribution(other_attr):
                composite_values = stack.get_base_values(other_attr, calculation_range)
                velocity = core.get_velocity_graph(composite_values)
            else:
                velocity = None

            if velocity and not core.is_equal(velocity):
                max_vel = max(velocity)
                if max_vel > best_max_vel:
                    best_max_vel = max_vel
                    best_velocity = velocity[:]
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


def _emit(processed, attr_data, new_values, context, apply):
    """Apply (or, in a dry run, just record) one attribute's new values."""
    if not apply:
        processed.append(attr_data.full_attr)
    elif apply_keyframes(attr_data, new_values, context):
        processed.append(attr_data.full_attr)


def _report(processed, apply, label):
    if apply:
        log.info("Adjusted attributes (%s): %s", label, processed)
    else:
        cmds.warning(
            f"apply=False: dry run, no keyframes written. Would adjust ({label}): {processed}"
        )
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

    processed: List[str] = []
    for attr_data in valid_attributes:
        new_values = calculate_adjustment_values(attr_data, context)
        _emit(processed, attr_data, new_values, context, apply)

    return _report(processed, apply, "scalar")


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
        try:
            stack = LayerStack.build(obj, context.adjustment_layer)
        except Exception as e:  # noqa: BLE001
            log.warning("Could not build LayerStack for %s: %s", obj, e)
            continue
        needed = set(core.CHANNELS) if smart else set(groups)
        obj_speeds[obj] = _group_speeds(obj, needed, stack, context.calculation_range)

    processed: List[str] = []
    left_sparse: List[str] = []
    for ad in attribute_data_list:
        speeds = obj_speeds.get(ad.obj)
        if not speeds:
            continue

        group = vector_core.channel_of(ad.attr)
        speed = speeds.get(group)
        source = group

        # Nothing on this group's own motion: optionally borrow another channel.
        if speed is None or core.is_equal(speed):
            borrow = vector_core.hottest_group(speeds, exclude=group) if smart else None
            if borrow:
                source, speed = borrow

        if speed is None or core.is_equal(speed):
            # No motion to ride anywhere — leave the attr sparse (Maya's own
            # linear blend already covers a motionless offset).
            left_sparse.append(ad.full_attr)
            continue

        if source != group:
            log.debug("Vector: %s riding %s-group speed", ad.attr, source)

        new_values = vector_core.distribute_by_speed(
            speed, ad.adjustment_values,
            context.adjustment_key_ranges, context.calculation_range,
        )
        _emit(processed, ad, new_values, context, apply)

    if left_sparse:
        log.debug("Vector: no motion to ride, left sparse: %s", left_sparse)

    return _report(processed, apply, "vector")


def _group_speeds(
    obj: str,
    groups: set,
    stack: LayerStack,
    calc_range: List[float],
) -> Dict[str, List[float]]:
    """Sample the below-motion for each axis of each group and reduce to speed."""
    speeds: Dict[str, List[float]] = {}
    n = len(calc_range)

    for group in groups:
        axes: Dict[str, List[float]] = {}
        for axis in ("X", "Y", "Z"):
            attr = group + axis
            if stack.ensure_contribution(attr):
                axes[axis] = stack.get_base_values(attr, calc_range)
            else:
                axes[axis] = [0.0] * n

        order = "xyz"
        if group == "rotate":
            try:
                order = cmds.getAttr(f"{obj}.rotateOrder")
            except Exception:  # noqa: BLE001 - fall back to a sane default order
                order = "xyz"

        speeds[group] = vector_core.channel_group_speed(group, axes, order)

    return speeds
