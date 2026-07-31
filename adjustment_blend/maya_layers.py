"""
LayerStack — discovers and queries Maya's animation layer structure.

This is the Maya-dependent machinery that walks the blend-node chains Maya
builds for layered animation and reports the *composite* value contributed by
every layer below a target layer. The adjustment-blend pipeline uses it to read
the motion underneath the adjustment layer without caring how Maya wired the
blend nodes together.

Extracted from the original ``animLib`` so the tool can ship standalone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from maya import cmds
import maya.api.OpenMaya as om
import maya.api.OpenMayaAnim as oma

from . import core


@dataclass
class _Contribution:
    """A single layer's contribution to an attribute."""
    layer: str
    curve_fn: Optional[oma.MFnAnimCurve] = None
    static_value: float = 0.0
    curve_name: str = ""


@dataclass
class _LayerInfo:
    """Information about a single animation layer."""
    name: str
    weight_curve_fn: Optional[oma.MFnAnimCurve] = None
    weight_static: float = 1.0
    muted: bool = False
    rotation_mode: int = 0  # 0=component, 1=quaternion
    scale_mode: int = 0  # 0=additive, 1=multiplicative


@dataclass
class LayerStack:
    """
    Discovers and queries the layer structure for a node.

    Handles:
    - Traversing blend node chains to find layer contributions
    - Getting composite values from layers below a target layer
    - Supporting animated layer weights
    """
    node: str
    target_layer: str
    layers: dict = field(default_factory=dict)
    layer_order: list = field(default_factory=list)
    contributions: dict = field(default_factory=dict)
    target_curves: dict = field(default_factory=dict)

    @classmethod
    def build(cls, node: str, target_layer: str) -> "LayerStack":
        """Build a LayerStack by traversing blend node chains for all layer attributes."""
        stack = cls(node=node, target_layer=target_layer)

        # Get target layer properties
        stack.layers[target_layer] = _LayerInfo(
            name=target_layer,
            weight_curve_fn=_get_weight_curve_fn(target_layer),
            weight_static=_get_static_weight(target_layer),
            muted=cmds.getAttr(f"{target_layer}.mute") if cmds.objExists(f"{target_layer}.mute") else False,
            rotation_mode=cmds.getAttr(f"{target_layer}.rotationAccumulationMode"),
            scale_mode=cmds.getAttr(f"{target_layer}.scaleAccumulationMode"),
        )

        # Build curve -> layer lookup
        curve_to_layer = {}
        all_layers = cmds.ls(type="animLayer") or []
        for layer in all_layers:
            layer_curves = cmds.animLayer(layer, query=True, animCurves=True) or []
            for curve in layer_curves:
                curve_to_layer[curve] = layer
            if layer != target_layer and layer not in stack.layers:
                stack.layers[layer] = _LayerInfo(
                    name=layer,
                    weight_curve_fn=_get_weight_curve_fn(layer),
                    weight_static=_get_static_weight(layer),
                    muted=cmds.getAttr(f"{layer}.mute") if cmds.objExists(f"{layer}.mute") else False,
                    rotation_mode=cmds.getAttr(f"{layer}.rotationAccumulationMode") if cmds.objExists(f"{layer}.rotationAccumulationMode") else 0,
                    scale_mode=cmds.getAttr(f"{layer}.scaleAccumulationMode") if cmds.objExists(f"{layer}.scaleAccumulationMode") else 0,
                )

        layer_curves = cmds.animLayer(target_layer, query=True, animCurves=True) or []
        target_curve_set = set(layer_curves)

        layer_attrs = cmds.animLayer(target_layer, query=True, attribute=True) or []
        node_attrs = [a.split(".")[-1] for a in layer_attrs if a.startswith(f"{node}.")]

        if not node_attrs:
            return stack

        for attr in node_attrs:
            attr_long = core.normalize_attr_name(attr)
            target_curve_fn = _find_target_curve(node, attr_long, target_layer, target_curve_set)
            if target_curve_fn:
                stack.target_curves[attr_long] = target_curve_fn

            contribs = _traverse_for_contributions(node, attr_long, target_layer, target_curve_set, stack, curve_to_layer)
            if contribs:
                stack.contributions[attr_long] = contribs

        seen_layers = set()
        for attr, contribs in stack.contributions.items():
            for c in contribs:
                if c.layer and c.layer != target_layer and c.layer not in seen_layers:
                    seen_layers.add(c.layer)
                    stack.layer_order.append(c.layer)

        return stack

    def get_base_value(self, attr: str, time: float) -> float:
        """Get composited base value for an attribute at a single time."""
        attr_long = core.normalize_attr_name(attr)
        contribs = self.contributions.get(attr_long, [])

        if not contribs:
            if attr_long in core.SCALE_ATTRS and self.layers[self.target_layer].scale_mode == 1:
                return 1.0
            return 0.0

        is_rotation = attr_long in core.ROTATION_ATTRS
        mtime = om.MTime(time, om.MTime.uiUnit())

        total = 0.0
        for contrib in contribs:
            weight = self.get_layer_weight(contrib.layer, time)
            if contrib.curve_fn:
                value = contrib.curve_fn.evaluate(mtime)
                if is_rotation:
                    value = math.degrees(value)
            else:
                value = contrib.static_value
            total += value * weight

        return total

    def get_base_values(self, attr: str, times: list) -> list:
        """Get composited base values for an attribute at multiple times."""
        return [self.get_base_value(attr, t) for t in times]

    def get_layer_weight(self, layer: str, time: float) -> float:
        """Get a layer's weight at a specific time."""
        if layer == "BaseAnimation" or layer is None:
            return 1.0

        layer_info = self.layers.get(layer)
        if not layer_info:
            if cmds.objExists(f"{layer}.weight"):
                return cmds.getAttr(f"{layer}.weight")
            return 1.0

        if layer_info.weight_curve_fn:
            mtime = om.MTime(time, om.MTime.uiUnit())
            return layer_info.weight_curve_fn.evaluate(mtime)

        return layer_info.weight_static

    def has_contribution(self, attr: str) -> bool:
        """Check if any layer below target contributes to this attr."""
        attr_long = core.normalize_attr_name(attr)
        return attr_long in self.contributions and len(self.contributions[attr_long]) > 0

    def ensure_contribution(self, attr: str) -> bool:
        """
        Build contribution data for an attribute if it doesn't already exist.

        This allows querying composite values for attributes that aren't on
        the target layer (e.g., for smart fallback).
        """
        attr_long = core.normalize_attr_name(attr)

        # Already have contributions for this attr
        if attr_long in self.contributions:
            return len(self.contributions[attr_long]) > 0

        # Build curve -> layer lookup if not cached
        if not hasattr(self, "_curve_to_layer"):
            self._curve_to_layer = {}
            all_layers = cmds.ls(type="animLayer") or []
            for layer in all_layers:
                layer_curves = cmds.animLayer(layer, query=True, animCurves=True) or []
                for curve in layer_curves:
                    self._curve_to_layer[curve] = layer

        # Get target layer curves for exclusion
        layer_curves = cmds.animLayer(self.target_layer, query=True, animCurves=True) or []
        target_curve_set = set(layer_curves)

        # Traverse blend node chain for this attribute
        contribs = _traverse_for_contributions(
            self.node, attr_long, self.target_layer,
            target_curve_set, self, self._curve_to_layer
        )

        if contribs:
            self.contributions[attr_long] = contribs
            return True

        return False


# =============================================================================
# LayerStack Helper Functions
# =============================================================================

def _get_weight_curve_fn(layer: str) -> Optional[oma.MFnAnimCurve]:
    """Get MFnAnimCurve for a layer's weight if animated."""
    weight_plug = f"{layer}.weight"
    if not cmds.objExists(weight_plug):
        return None

    curves = cmds.listConnections(weight_plug, source=True, destination=False, type="animCurve")
    if curves:
        sel = om.MSelectionList()
        sel.add(curves[0])
        return oma.MFnAnimCurve(sel.getDependNode(0))
    return None


def _get_static_weight(layer: str) -> float:
    """Get static weight value for a layer."""
    weight_plug = f"{layer}.weight"
    if cmds.objExists(weight_plug):
        return cmds.getAttr(weight_plug)
    return 1.0


def _find_target_curve(node: str, attr: str, target_layer: str,
                       target_curve_set: set) -> Optional[oma.MFnAnimCurve]:
    """Find the target layer's curve for a specific attribute."""
    for curve_name in target_curve_set:
        outputs = cmds.listConnections(f"{curve_name}.output", source=False,
                                       destination=True, plugs=True) or []
        for out in outputs:
            target = _trace_to_attribute(out)
            if target and target[0] == node and core.normalize_attr_name(target[1]) == attr:
                sel = om.MSelectionList()
                sel.add(curve_name)
                return oma.MFnAnimCurve(sel.getDependNode(0))
    return None


def _trace_to_attribute(plug: str, max_depth: int = 10) -> Optional[tuple]:
    """Trace from a plug through blend nodes to find the final node.attr."""
    visited = set()
    current = plug

    for _ in range(max_depth):
        if not current or current in visited:
            break
        visited.add(current)

        node = current.split(".")[0]
        node_type = cmds.nodeType(node)

        if node_type in ("transform", "joint", "ikHandle") or cmds.objectType(node, isAType="transform"):
            attr = current.split(".")[-1]
            return (node, attr)

        if "Blend" in node_type or "blend" in node_type:
            out_attr = "output"
            if "X" in current.split(".")[-1]:
                out_attr = "outputX"
            elif "Y" in current.split(".")[-1]:
                out_attr = "outputY"
            elif "Z" in current.split(".")[-1]:
                out_attr = "outputZ"

            out_plug = f"{node}.{out_attr}"
            if cmds.objExists(out_plug):
                outputs = cmds.listConnections(out_plug, source=False,
                                               destination=True, plugs=True) or []
                if outputs:
                    current = outputs[0]
                    continue

        break

    return None


def _traverse_for_contributions(node: str, attr: str, target_layer: str,
                                target_curve_set: set, stack: "LayerStack",
                                curve_to_layer: dict) -> list:
    """Traverse blend node chain to find all contributions below the target layer."""
    contributions = []
    plug = f"{node}.{attr}"

    conns = cmds.listConnections(plug, source=True, destination=False,
                                 plugs=True, skipConversionNodes=True) or []

    blend_node = None
    for conn in conns:
        conn_node = conn.split(".")[0]
        conn_type = cmds.nodeType(conn_node)

        if conn_type == "pairBlend":
            return []

        if "animBlend" in conn_type or conn_type == "animBlendNodeBase":
            blend_node = conn_node
            break

    if not blend_node:
        return contributions

    visited = set()
    current_blend = blend_node

    while current_blend and current_blend not in visited:
        visited.add(current_blend)

        input_b_name = _get_input_b_name(current_blend, attr)
        input_b_plug = f"{current_blend}.{input_b_name}"

        input_b_conns = cmds.listConnections(input_b_plug, source=True, destination=False,
                                             plugs=True, skipConversionNodes=True) or []

        curve_fn = None
        static_value = 0.0
        source_layer = None
        curve_name = ""
        is_target_layer = False

        if input_b_conns:
            for conn in input_b_conns:
                conn_node = conn.split(".")[0]
                conn_type = cmds.nodeType(conn_node)

                if conn_type.startswith("animCurve"):
                    curve_name = conn_node

                    if conn_node in target_curve_set:
                        is_target_layer = True
                        break

                    sel = om.MSelectionList()
                    sel.add(conn_node)
                    curve_fn = oma.MFnAnimCurve(sel.getDependNode(0))
                    source_layer = curve_to_layer.get(conn_node)
                    break
        else:
            static_value = cmds.getAttr(input_b_plug)

        if not is_target_layer and (curve_fn or input_b_conns == []):
            contributions.append(_Contribution(
                layer=source_layer or "BaseAnimation",
                curve_fn=curve_fn,
                static_value=static_value,
                curve_name=curve_name,
            ))

        input_a_name = _get_input_a_name(current_blend, attr)
        input_a_plug = f"{current_blend}.{input_a_name}"

        input_a_conns = cmds.listConnections(input_a_plug, source=True, destination=False,
                                             skipConversionNodes=True) or []

        current_blend = None
        for conn in input_a_conns:
            conn_type = cmds.nodeType(conn)
            if "animBlend" in conn_type or conn_type == "animBlendNodeBase":
                current_blend = conn
                break
            elif conn_type.startswith("animCurve"):
                layer_name = curve_to_layer.get(conn)
                is_base = not layer_name or layer_name == "BaseAnimation"

                if is_base:
                    sel = om.MSelectionList()
                    sel.add(conn)
                    base_curve_fn = oma.MFnAnimCurve(sel.getDependNode(0))
                    contributions.append(_Contribution(
                        layer="BaseAnimation",
                        curve_fn=base_curve_fn,
                        static_value=0.0,
                        curve_name=conn,
                    ))
                break

    contributions.reverse()
    return contributions


def _get_input_a_name(blend_node: str, attr: str) -> str:
    """Get correct inputA plug name based on blend node type and attribute."""
    blend_type = cmds.nodeType(blend_node)
    attr_long = core.normalize_attr_name(attr)

    if blend_type == "animBlendNodeAdditiveRotation":
        if attr_long == "rotateX":
            return "inputAX"
        elif attr_long == "rotateY":
            return "inputAY"
        elif attr_long == "rotateZ":
            return "inputAZ"

    return "inputA"


def _get_input_b_name(blend_node: str, attr: str) -> str:
    """Get correct inputB plug name based on blend node type and attribute."""
    blend_type = cmds.nodeType(blend_node)
    attr_long = core.normalize_attr_name(attr)

    if blend_type == "animBlendNodeAdditiveRotation":
        if attr_long == "rotateX":
            return "inputBX"
        elif attr_long == "rotateY":
            return "inputBY"
        elif attr_long == "rotateZ":
            return "inputBZ"

    return "inputB"
