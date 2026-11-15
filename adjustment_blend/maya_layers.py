"""
LayerStack — reads the motion underneath an animation layer, fast.

How Maya composites layers
--------------------------
Every layered attribute is driven by a chain of ``animBlendNodeAdditive*``
nodes, one per layer, with the lowest layer deepest in the chain::

    ctrl.tx <- blend (top layer) <- ... <- blend (lowest layer) <- base curve
                 inputB <- that layer's own curve
                 inputA <- everything below that layer

Each blend node computes ``inputA * weightA + inputB * weightB`` (a scale blend
in multiply mode computes ``inputA ** weightA * inputB ** weightB``). The two
weights come from the layer's ``backgroundWeight`` / ``foregroundWeight``,
which already account for the layer's weight, mute, solo, override and
passthrough settings.

So "everything below layer L" is simply the ``inputA`` of L's blend node, and
it can be reproduced by evaluating the anim curves and doing that arithmetic
here. That is about 100x faster than asking Maya to evaluate the plug at every
frame, and it matches Maya's result exactly (checked against Maya 2025).

Where arithmetic cannot follow — a layer that accumulates rotation as
quaternions, or an input driven by something other than a curve — the plug is
evaluated by Maya instead. That is correct but slow.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from maya import cmds
import maya.api.OpenMaya as om
import maya.api.OpenMayaAnim as oma

_BLEND_PREFIX = "animBlendNodeAdditive"
_ROTATION_BLEND = "animBlendNodeAdditiveRotation"
_SCALE_BLEND = "animBlendNodeAdditiveScale"

# ``accumulationMode`` values on the rotation / scale blend nodes.
_QUATERNION = 1
_MULTIPLY = 1

# Layer attributes that make foregroundWeight / backgroundWeight change over
# time when they are animated.
_WEIGHT_DRIVERS = ("weight", "mute", "solo", "override")


class _NeedsMaya(Exception):
    """The chain contains something curve math cannot reproduce."""


class LayerStack:
    """Samples the composite animation below ``target_layer``.

    Build one per run and share it across every object: layer weights and
    results are cached, so asking twice for the same attribute is free.
    """

    def __init__(self, target_layer: str, times: Sequence[float]):
        self.target_layer = target_layer
        self.times = list(times)
        unit = om.MTime.uiUnit()
        self._mtimes = [om.MTime(t, unit) for t in self.times]
        self._nodes: Dict[str, Optional[om.MFnDependencyNode]] = {}
        self._weights: Dict[str, Tuple[List[float], List[float], bool]] = {}
        self._values: Dict[Tuple[str, str], Optional[List[float]]] = {}

    def base_values(self, node: str, attr: str) -> Optional[List[float]]:
        """Return the value below the target layer at each sample time.

        Rotations are in degrees. For an attribute that is not a member of the
        target layer, the whole layered value is returned. Returns ``None``
        when the attribute has no animation to read (static, missing, or
        driven by a constraint or expression).

        The list is cached and shared: do not modify it.
        """
        key = (node, attr)
        if key not in self._values:
            self._values[key] = self._read(node, attr)
        return self._values[key]

    def curve_values(self, curve: str) -> List[float]:
        """Return an anim curve's value at each sample time, in UI units.

        The same numbers as ``cmds.keyframe(curve, q=True, eval=True)`` asked
        frame by frame, without a command call per frame.
        """
        selection = om.MSelectionList()
        selection.add(curve)
        curve_fn = oma.MFnAnimCurve(selection.getDependNode(0))
        evaluate = curve_fn.evaluate
        to_ui = to_ui_units(curve_fn.animCurveType)
        return [evaluate(t) * to_ui for t in self._mtimes]

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    def _read(self, node: str, attr: str) -> Optional[List[float]]:
        plug = self._plug(node, attr)
        if plug is None:
            return None

        source = plug.source()
        if source.isNull:
            return None
        if not (source.node().hasFn(om.MFn.kAnimCurve) or _is_blend(source)):
            return None

        below = self._below_target(plug)
        try:
            values = self._series(below)
        except _NeedsMaya:
            values = self._evaluate(below)

        if _is_angle(plug):
            values = [math.degrees(v) for v in values]
        return values

    def _plug(self, node: str, attr: str) -> Optional[om.MPlug]:
        if node not in self._nodes:
            try:
                selection = om.MSelectionList()
                selection.add(node)
                self._nodes[node] = om.MFnDependencyNode(selection.getDependNode(0))
            except RuntimeError:
                self._nodes[node] = None

        node_fn = self._nodes[node]
        if node_fn is None:
            return None
        try:
            return node_fn.findPlug(attr, False)
        except RuntimeError:
            return None

    def _below_target(self, plug: om.MPlug) -> om.MPlug:
        """Return the plug that carries everything below the target layer.

        That is the ``inputA`` of the target layer's blend node. If the target
        layer is not in this attribute's chain, the attribute's own plug is
        returned (the whole chain).
        """
        current = plug
        while True:
            source = current.source()
            if source.isNull or not _is_blend(source):
                return plug
            blend_fn = om.MFnDependencyNode(source.node())
            input_a = blend_fn.findPlug("inputA" + _axis_suffix(source), False)
            if _layer_name(blend_fn) == self.target_layer:
                return input_a
            current = input_a

    def _series(self, plug: om.MPlug) -> List[float]:
        """Value of an input plug at every sample time, by curve math."""
        source = plug.source()
        if source.isNull:
            return [plug.asDouble()] * len(self._mtimes)

        source_node = source.node()
        if source_node.hasFn(om.MFn.kAnimCurve):
            evaluate = oma.MFnAnimCurve(source_node).evaluate
            return [evaluate(t) for t in self._mtimes]

        if not _is_blend(source):
            raise _NeedsMaya

        blend_fn = om.MFnDependencyNode(source_node)
        blend_type = blend_fn.typeName
        suffix = _axis_suffix(source)

        multiply = False
        if blend_type == _ROTATION_BLEND:
            if blend_fn.findPlug("accumulationMode", False).asInt() == _QUATERNION:
                raise _NeedsMaya
        elif blend_type == _SCALE_BLEND:
            multiply = blend_fn.findPlug("accumulationMode", False).asInt() == _MULTIPLY

        weight_a, weight_b, plain = self._layer_weights(blend_fn)
        a = self._series(blend_fn.findPlug("inputA" + suffix, False))
        b = self._series(blend_fn.findPlug("inputB" + suffix, False))

        if multiply:
            if plain:
                return [x * y for x, y in zip(a, b)]
            if min(a) < 0.0 or min(b) < 0.0:
                raise _NeedsMaya  # fractional power of a negative scale
            return [x ** p * y ** q for x, p, y, q in zip(a, weight_a, b, weight_b)]
        if plain:
            return [x + y for x, y in zip(a, b)]
        return [x * p + y * q for x, p, y, q in zip(a, weight_a, b, weight_b)]

    def _layer_weights(self, blend_fn: om.MFnDependencyNode) -> Tuple[List[float], List[float], bool]:
        """Return ``(weightA per frame, weightB per frame, both are all 1.0)``.

        Cached per layer: every blend node of a layer shares the same weights.
        """
        background = blend_fn.findPlug("weightA", False).source()
        foreground = blend_fn.findPlug("weightB", False).source()
        if background.isNull or foreground.isNull:
            raise _NeedsMaya

        layer = om.MFnDependencyNode(foreground.node()).name()
        if layer not in self._weights:
            if _weights_are_animated(layer):
                weight_a = self._evaluate(background)
                weight_b = self._evaluate(foreground)
            else:
                count = len(self._mtimes)
                weight_a = [background.asDouble()] * count
                weight_b = [foreground.asDouble()] * count
            plain = all(w == 1.0 for w in weight_a) and all(w == 1.0 for w in weight_b)
            self._weights[layer] = (weight_a, weight_b, plain)
        return self._weights[layer]

    def _evaluate(self, plug: om.MPlug) -> List[float]:
        """Value of a plug at every sample time, evaluated by Maya (slow)."""
        values = []
        for mtime in self._mtimes:
            # Keep a reference: Maya crashes if the current context is freed.
            context = om.MDGContext(mtime)
            previous = context.makeCurrent()
            try:
                values.append(plug.asDouble())
            finally:
                previous.makeCurrent()
        return values


# =============================================================================
# Helpers
# =============================================================================

def _is_blend(plug: om.MPlug) -> bool:
    return om.MFnDependencyNode(plug.node()).typeName.startswith(_BLEND_PREFIX)


def _axis_suffix(output_plug: om.MPlug) -> str:
    """``outputX`` -> ``X``; a single-channel ``output`` -> empty string."""
    return output_plug.partialName(useLongNames=True)[len("output"):]


def _layer_name(blend_fn: om.MFnDependencyNode) -> Optional[str]:
    """Name of the anim layer that owns a blend node."""
    foreground = blend_fn.findPlug("weightB", False).source()
    if foreground.isNull:
        return None
    return om.MFnDependencyNode(foreground.node()).name()


def _is_angle(plug: om.MPlug) -> bool:
    attribute = plug.attribute()
    return (attribute.hasFn(om.MFn.kUnitAttribute)
            and om.MFnUnitAttribute(attribute).unitType() == om.MFnUnitAttribute.kAngle)


def to_ui_units(curve_type: int) -> float:
    """Factor from a curve's internal units (radians, centimeters) to UI units."""
    if curve_type == oma.MFnAnimCurve.kAnimCurveTA:
        return om.MAngle(1.0, om.MAngle.kRadians).asUnits(om.MAngle.uiUnit())
    if curve_type == oma.MFnAnimCurve.kAnimCurveTL:
        return om.MDistance(1.0, om.MDistance.kCentimeters).asUnits(om.MDistance.uiUnit())
    return 1.0


def _weights_are_animated(layer: str) -> bool:
    """True if this layer's weights can change over time.

    A parent layer's weight and mute feed into its children, so the layer's
    ancestors are checked too.
    """
    while layer:
        for attr in _WEIGHT_DRIVERS:
            if cmds.listConnections(f"{layer}.{attr}", source=True, destination=False):
                return True
        layer = cmds.animLayer(layer, query=True, parent=True)
    return False
