"""
AnimLib - Core animation library functions for Maya.

This module provides the core functionality for:
- Animation pin creation and baking
- Override-to-additive layer conversion
- Transform extraction and constraint management
- Euler filtering and quaternion math

This is the "engine" that powers the Animation Pin Tool UI.
"""

__author__ = "Daniel Klug"
__version__ = "2.0.0"
__date__ = "2026-01-01"
__email__ = "daniel@redforty.com"

# =============================================================================
# Imports
# =============================================================================

import math
import time
import itertools
import re
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import wraps
from typing import Optional, List, Dict, Callable

import maya.cmds as cmds
import maya.mel as mel
import maya.api.OpenMaya as api
import maya.api.OpenMayaAnim as anim
import numpy as np

# =============================================================================
# Logging Setup
# =============================================================================

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)

# =============================================================================
# Constants and Globals
# =============================================================================

# Pin tool constants
QUICKSELECTSET = 'animPins'
AP_SUFFIX = '_pin'
PIN_GROUP = 'pin_group#'
MASTER_GROUP = 'animPins_group'
LOCATOR_SCALE = 10  # Season to taste - depends on your rig size
LOCATOR_COLOR = (0.273, 0.432, 0.152)  # Light green

# Pin data structure template
PIN_DATA = OrderedDict([
    ('control', {'dataType': "string"}),
    ('constraint', {'dataType': "string"}),
    ('start_frame', {'attributeType': "float"}),
    ('end_frame', {'attributeType': "float"}),
    ('translate_keys', {'dataType': "string"}),
    ('rotate_keys', {'dataType': "string"}),
    ('translate_locked', {'attributeType': "bool"}),
    ('rotate_locked', {'attributeType': "bool"}),
    ('preserve_blendParent', {'dataType': "string"})
])

# Get the timeline object
aPlayBackSliderPython = mel.eval('$tmpVar=$gPlayBackSlider')
tuple_rex = re.compile(r"([0-9]+\.[0-9]+\, [0-9]+\.[0-9]+)")

# Attribute mappings for layer conversion
SHORT_TO_LONG = {
    'tx': 'translateX', 'ty': 'translateY', 'tz': 'translateZ',
    'rx': 'rotateX', 'ry': 'rotateY', 'rz': 'rotateZ',
    'sx': 'scaleX', 'sy': 'scaleY', 'sz': 'scaleZ'
}
LONG_TO_SHORT = {v: k for k, v in SHORT_TO_LONG.items()}

ROTATION_ATTRS = {'rotateX', 'rotateY', 'rotateZ', 'rx', 'ry', 'rz'}
SCALE_ATTRS = {'scaleX', 'scaleY', 'scaleZ', 'sx', 'sy', 'sz'}
TRANSLATE_ATTRS = {'translateX', 'translateY', 'translateZ', 'tx', 'ty', 'tz'}
TRANSFORM_ATTRS = TRANSLATE_ATTRS | ROTATION_ATTRS | SCALE_ATTRS


# =============================================================================
# Decorators
# =============================================================================

def viewport_off(func):
    """
    Decorator - turn off Maya display while func is running.
    If func fails, the error will be raised after viewport is restored.
    """
    @wraps(func)
    def wrap(*args, **kwargs):
        parallel = False
        if 'parallel' in cmds.evaluationManager(q=True, mode=True):
            cmds.evaluationManager(mode='off')
            parallel = True

        # Turn $gMainPane Off:
        mel.eval("paneLayout -e -manage false $gMainPane")
        cmds.refresh(suspend=True)
        # Hide the timeslider
        mel.eval("setTimeSliderVisible 0;")

        try:
            return func(*args, **kwargs)
        except Exception:
            raise
        finally:
            cmds.refresh(suspend=False)
            mel.eval("setTimeSliderVisible 1;")
            if parallel:
                cmds.evaluationManager(mode='parallel')
            mel.eval("paneLayout -e -manage true $gMainPane")

    return wrap


def undo(func):
    """Decorator - open/close undo chunk"""
    @wraps(func)
    def wrap(*args, **kwargs):
        cmds.undoInfo(openChunk=True)
        try:
            return func(*args, **kwargs)
        except Exception:
            raise
        finally:
            cmds.undoInfo(closeChunk=True)

    return wrap


def noUndo(func):
    """Decorator - disable undo for this function"""
    @wraps(func)
    def wrap(*args, **kwargs):
        cmds.undoInfo(stateWithoutFlush=False)
        try:
            return func(*args, **kwargs)
        except Exception:
            raise
        finally:
            cmds.undoInfo(stateWithoutFlush=True)

    return wrap


# =============================================================================
# Layer Stack Data Structures
# =============================================================================

@dataclass
class Contribution:
    """A single layer's contribution to an attribute."""
    layer: str
    curve_fn: Optional[anim.MFnAnimCurve] = None
    static_value: float = 0.0
    curve_name: str = ""


@dataclass
class LayerInfo:
    """Information about a single animation layer."""
    name: str
    weight_curve_fn: Optional[anim.MFnAnimCurve] = None
    weight_static: float = 1.0
    muted: bool = False
    rotation_mode: int = 0  # 0=component, 1=quaternion
    scale_mode: int = 0  # 0=additive, 1=multiplicative


@dataclass
class LayerStack:
    """
    Discovered layer structure for a node, used during override-to-additive conversion.
    """
    node: str
    target_layer: str
    layers: dict = field(default_factory=dict)
    layer_order: list = field(default_factory=list)
    contributions: dict = field(default_factory=dict)
    target_curves: dict = field(default_factory=dict)

    @classmethod
    def build(cls, node: str, target_layer: str) -> 'LayerStack':
        """Build a LayerStack by traversing blend node chains for all layer attributes."""
        stack = cls(node=node, target_layer=target_layer)

        # Get target layer properties
        stack.layers[target_layer] = LayerInfo(
            name=target_layer,
            weight_curve_fn=_get_weight_curve_fn(target_layer),
            weight_static=_get_static_weight(target_layer),
            muted=cmds.getAttr(f"{target_layer}.mute") if cmds.objExists(f"{target_layer}.mute") else False,
            rotation_mode=cmds.getAttr(f"{target_layer}.rotationAccumulationMode"),
            scale_mode=cmds.getAttr(f"{target_layer}.scaleAccumulationMode"),
        )

        # Build curve -> layer lookup
        curve_to_layer = {}
        all_layers = cmds.ls(type='animLayer') or []
        for layer in all_layers:
            layer_curves = cmds.animLayer(layer, query=True, animCurves=True) or []
            for curve in layer_curves:
                curve_to_layer[curve] = layer
            if layer != target_layer and layer not in stack.layers:
                stack.layers[layer] = LayerInfo(
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
        node_attrs = [a.split('.')[-1] for a in layer_attrs if a.startswith(f"{node}.")]

        if not node_attrs:
            return stack

        for attr in node_attrs:
            attr_long = normalizeAttrName(attr)
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
        attr_long = normalizeAttrName(attr)
        contribs = self.contributions.get(attr_long, [])

        if not contribs:
            if attr_long in SCALE_ATTRS and self.layers[self.target_layer].scale_mode == 1:
                return 1.0
            return 0.0

        is_rotation = attr_long in ROTATION_ATTRS
        mtime = api.MTime(time, api.MTime.uiUnit())

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
        if layer == 'BaseAnimation' or layer is None:
            return 1.0

        layer_info = self.layers.get(layer)
        if not layer_info:
            if cmds.objExists(f"{layer}.weight"):
                return cmds.getAttr(f"{layer}.weight")
            return 1.0

        if layer_info.weight_curve_fn:
            mtime = api.MTime(time, api.MTime.uiUnit())
            return layer_info.weight_curve_fn.evaluate(mtime)

        return layer_info.weight_static

    def has_contribution(self, attr: str) -> bool:
        """Check if any layer below target contributes to this attr."""
        attr_long = normalizeAttrName(attr)
        return attr_long in self.contributions and len(self.contributions[attr_long]) > 0

    def ensure_contribution(self, attr: str) -> bool:
        """
        Build contribution data for an attribute if it doesn't already exist.

        This allows querying composite values for attributes that aren't on
        the target layer (e.g., for smart fallback in adjustment_blend).

        Args:
            attr: Attribute name (e.g., 'translateY')

        Returns:
            True if contributions exist or were built, False if attribute
            has no animation data.
        """
        attr_long = normalizeAttrName(attr)

        # Already have contributions for this attr
        if attr_long in self.contributions:
            return len(self.contributions[attr_long]) > 0

        # Build curve -> layer lookup if not cached
        if not hasattr(self, '_curve_to_layer'):
            self._curve_to_layer = {}
            all_layers = cmds.ls(type='animLayer') or []
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

    def diagnose(self) -> str:
        """Generate a diagnostic string showing the discovered layer structure."""
        lines = [
            f"LayerStack for '{self.node}' (target: {self.target_layer})",
            "=" * 60,
            f"Layer order (bottom to top): {self.layer_order}",
            "",
            f"Target curves ({len(self.target_curves)}):",
        ]

        for attr, curve_fn in sorted(self.target_curves.items()):
            num_keys = curve_fn.numKeys if curve_fn else 0
            lines.append(f"  {attr}: {num_keys} keys")

        lines.append("")
        lines.append("Contributions by attribute:")

        for attr, contribs in sorted(self.contributions.items()):
            lines.append(f"  {attr}:")
            for c in contribs:
                if c.curve_fn:
                    lines.append(f"    - {c.layer}: curve '{c.curve_name}' ({c.curve_fn.numKeys} keys)")
                else:
                    lines.append(f"    - {c.layer}: static = {c.static_value}")

        return "\n".join(lines)


# =============================================================================
# Helper Functions for LayerStack
# =============================================================================

def _get_weight_curve_fn(layer: str) -> Optional[anim.MFnAnimCurve]:
    """Get MFnAnimCurve for a layer's weight if animated."""
    weight_plug = f"{layer}.weight"
    if not cmds.objExists(weight_plug):
        return None

    curves = cmds.listConnections(weight_plug, source=True, destination=False, type='animCurve')
    if curves:
        sel = api.MSelectionList()
        sel.add(curves[0])
        return anim.MFnAnimCurve(sel.getDependNode(0))
    return None


def _get_static_weight(layer: str) -> float:
    """Get static weight value for a layer."""
    weight_plug = f"{layer}.weight"
    if cmds.objExists(weight_plug):
        return cmds.getAttr(weight_plug)
    return 1.0


def _find_target_curve(node: str, attr: str, target_layer: str,
                       target_curve_set: set) -> Optional[anim.MFnAnimCurve]:
    """Find the target layer's curve for a specific attribute."""
    for curve_name in target_curve_set:
        outputs = cmds.listConnections(f"{curve_name}.output", source=False,
                                        destination=True, plugs=True) or []
        for out in outputs:
            target = _trace_to_attribute(out)
            if target and target[0] == node and normalizeAttrName(target[1]) == attr:
                sel = api.MSelectionList()
                sel.add(curve_name)
                return anim.MFnAnimCurve(sel.getDependNode(0))
    return None


def _trace_to_attribute(plug: str, max_depth: int = 10) -> Optional[tuple]:
    """Trace from a plug through blend nodes to find the final node.attr."""
    visited = set()
    current = plug

    for _ in range(max_depth):
        if not current or current in visited:
            break
        visited.add(current)

        node = current.split('.')[0]
        node_type = cmds.nodeType(node)

        if node_type in ('transform', 'joint', 'ikHandle') or cmds.objectType(node, isAType='transform'):
            attr = current.split('.')[-1]
            return (node, attr)

        if 'Blend' in node_type or 'blend' in node_type:
            out_attr = 'output'
            if 'X' in current.split('.')[-1]:
                out_attr = 'outputX'
            elif 'Y' in current.split('.')[-1]:
                out_attr = 'outputY'
            elif 'Z' in current.split('.')[-1]:
                out_attr = 'outputZ'

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
                                target_curve_set: set, stack: 'LayerStack',
                                curve_to_layer: dict) -> list:
    """Traverse blend node chain to find all contributions below the target layer."""
    contributions = []
    plug = f"{node}.{attr}"

    conns = cmds.listConnections(plug, source=True, destination=False,
                                  plugs=True, skipConversionNodes=True) or []

    blend_node = None
    for conn in conns:
        conn_node = conn.split('.')[0]
        conn_type = cmds.nodeType(conn_node)

        if conn_type == 'pairBlend':
            return []

        if 'animBlend' in conn_type or conn_type == 'animBlendNodeBase':
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
                conn_node = conn.split('.')[0]
                conn_type = cmds.nodeType(conn_node)

                if conn_type.startswith('animCurve'):
                    curve_name = conn_node

                    if conn_node in target_curve_set:
                        is_target_layer = True
                        break

                    sel = api.MSelectionList()
                    sel.add(conn_node)
                    curve_fn = anim.MFnAnimCurve(sel.getDependNode(0))
                    source_layer = curve_to_layer.get(conn_node)
                    break
        else:
            static_value = cmds.getAttr(input_b_plug)

        if not is_target_layer and (curve_fn or input_b_conns == []):
            contributions.append(Contribution(
                layer=source_layer or 'BaseAnimation',
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
            if 'animBlend' in conn_type or conn_type == 'animBlendNodeBase':
                current_blend = conn
                break
            elif conn_type.startswith('animCurve'):
                layer_name = curve_to_layer.get(conn)
                is_base = not layer_name or layer_name == 'BaseAnimation'

                if is_base:
                    sel = api.MSelectionList()
                    sel.add(conn)
                    base_curve_fn = anim.MFnAnimCurve(sel.getDependNode(0))
                    contributions.append(Contribution(
                        layer='BaseAnimation',
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
    attr_long = normalizeAttrName(attr)

    if blend_type == "animBlendNodeAdditiveRotation":
        if attr_long == 'rotateX':
            return "inputAX"
        elif attr_long == 'rotateY':
            return "inputAY"
        elif attr_long == 'rotateZ':
            return "inputAZ"

    return "inputA"


def _get_input_b_name(blend_node: str, attr: str) -> str:
    """Get correct inputB plug name based on blend node type and attribute."""
    blend_type = cmds.nodeType(blend_node)
    attr_long = normalizeAttrName(attr)

    if blend_type == "animBlendNodeAdditiveRotation":
        if attr_long == 'rotateX':
            return "inputBX"
        elif attr_long == 'rotateY':
            return "inputBY"
        elif attr_long == 'rotateZ':
            return "inputBZ"

    return "inputB"


# =============================================================================
# Attribute Utilities
# =============================================================================

def normalizeAttrName(attr):
    """Convert attribute name to long form."""
    return SHORT_TO_LONG.get(attr, attr)


def getAttrAxis(attr):
    """Get the axis (X, Y, Z) from an attribute name."""
    attr_long = normalizeAttrName(attr)
    if attr_long.endswith('X'):
        return 'X'
    elif attr_long.endswith('Y'):
        return 'Y'
    elif attr_long.endswith('Z'):
        return 'Z'
    return None


# =============================================================================
# Pin Management Functions
# =============================================================================

def get_pin_groups():
    """Get all pin groups under the master group."""
    global MASTER_GROUP
    found_pin_groups = []
    if cmds.objExists(MASTER_GROUP):
        master_group_children = cmds.listRelatives(MASTER_GROUP, type='transform') or []
        for found_pin_group in master_group_children:
            found_pin_groups.append(found_pin_group)
    return found_pin_groups


def get_pins(pin_groups=None):
    """Get all pins in the specified pin groups."""
    if isinstance(pin_groups, str):
        pin_groups = [pin_groups]
    found_pins = []
    if not pin_groups:
        pin_groups = get_pin_groups()
    for group in pin_groups:
        children = cmds.listRelatives(group, type='transform') or []
        for child in children:
            if cmds.attributeQuery('control', node=child, exists=True):
                found_pins.append(child)
            else:
                grandchildren = cmds.listRelatives(child, type='transform') or []
                for grandchild in grandchildren:
                    if cmds.attributeQuery('control', node=grandchild, exists=True):
                        found_pins.append(grandchild)
    return found_pins


def get_master_group(group_override=None):
    """Get or create the master pin group."""
    global MASTER_GROUP
    if isinstance(group_override, str):
        MASTER_GROUP = group_override
    if not cmds.objExists(MASTER_GROUP):
        MASTER_GROUP = cmds.createNode('transform',
                                       name=MASTER_GROUP,
                                       skipSelect=True)
    return MASTER_GROUP


def create_new_pin_group():
    """Create a new pin group under the master group."""
    master_group = get_master_group()
    new_pin_group = cmds.createNode('transform',
                                    name=PIN_GROUP,
                                    parent=master_group,
                                    skipSelect=True)
    return new_pin_group


def get_selectionList(selection):
    """Convert selection to MSelectionList."""
    if not selection:
        return api.MGlobal.getActiveSelectionList()
    elif isinstance(selection, str):
        return api.MGlobal.getSelectionListByName(selection)
    elif isinstance(selection, list):
        nodes = api.MSelectionList()
        for sel in selection:
            try:
                node = api.MGlobal.getSelectionListByName(sel)
                nodes.merge(node)
            except:
                api.MGlobal.displayError(
                    "Could not fetch selection. "
                    "Try submitting an MSelectionList "
                    "or a list of string names."
                )
        return nodes
    else:
        return selection


def create_locator_pin(control_data, pin_group):
    """Create a locator pin with control data attributes.

    The control name stored in control_data['control'] may be a long DAG path
    (e.g., '|group1|arm_ctrl'). We extract the short name for the locator
    naming (since Maya node names cannot contain pipes), but store the full
    long name in the 'control' attribute for unique identification.
    """
    control_long_name = control_data['control']
    # Extract short name for locator naming (Maya doesn't allow pipes in names)
    control_short_name = get_short_name(control_long_name)

    locator = cmds.spaceLocator(name=control_short_name + AP_SUFFIX)[0]
    cmds.setAttr(locator + ".scale", *[LOCATOR_SCALE] * 3)
    # Parent returns the new long path - use it to avoid ambiguity with duplicate names
    locator = cmds.parent(locator, pin_group)[0]

    for attr in ['x', 'y', 'z']:
        cmds.setAttr(locator + '.s' + attr, keyable=False, channelBox=True)

    for key, value in PIN_DATA.items():
        cmds.addAttr(locator, longName=key, **value)

    for key, value in control_data.items():
        if isinstance(value, (str, list)):
            if isinstance(value, list):
                value = ' '.join(map(str, value))
            cmds.setAttr(locator + '.' + key, value, type='string')
        else:
            cmds.setAttr(locator + '.' + key, value)

    return locator


# =============================================================================
# Validation Functions
# =============================================================================

def validate_selection(sel_list):
    """Validate and filter selection for valid transform nodes."""
    current_pins = get_pins()
    validated_sel_list = api.MSelectionList()

    # Build a set of long names for controls that are already pinned
    pinned_control_long_names = set()
    pin_long_names = set()
    for pin in current_pins:
        # Get the stored control long name from the pin's attribute
        if cmds.attributeQuery('control', node=pin, exists=True):
            stored_control = cmds.getAttr(pin + '.control')
            if stored_control:
                pinned_control_long_names.add(stored_control)
        # Also get the pin's own long name
        pin_long_names.add(cmds.ls(pin, long=True)[0])

    for i in range(sel_list.length()):
        try:
            dag = sel_list.getDagPath(i)
        except TypeError:
            continue

        control_dep = sel_list.getDependNode(i)
        controlFN = api.MFnDependencyNode(control_dep)
        # Use long name for unique identification
        control_long_name = dag.fullPathName()
        control_short_name = get_short_name(control_long_name)

        # Check if this control is already pinned (using long name comparison)
        if control_long_name in pinned_control_long_names:
            api.MGlobal.displayError(
                "Node '%s' is already pinned! Skipping..." % control_short_name)
            continue

        # Check if user selected a pin itself (using long name comparison)
        if control_long_name in pin_long_names:
            api.MGlobal.displayError(
                "Node '%s' is a pin! Skipping..." % control_short_name)
            continue

        if controlFN.typeName not in ['transform', 'joint', 'ikHandle']:
            api.MGlobal.displayError(
                "Node '%s' is not a valid transform node. Skipping..." % control_short_name)
            continue

        if is_locked_or_not_keyable(controlFN, 'translate') and \
           is_locked_or_not_keyable(controlFN, 'rotate'):
            api.MGlobal.displayError(
                "Node '%s' has no available transform channels. Skipping..." % control_short_name)
            continue

        validated_sel_list.add(sel_list.getDependNode(i))

    return validated_sel_list


def validate_framerange(start_frame, end_frame):
    """Validate and return frame range."""
    if start_frame is None:
        start_frame = cmds.playbackOptions(query=True, minTime=True)

    if end_frame is None:
        end_frame = cmds.playbackOptions(query=True, maxTime=True)

    if start_frame > end_frame:
        api.MGlobal.displayError("Start frame needs to be before end frame!")
        return None, None

    return start_frame, end_frame


def validate_bakerange(pins_to_bake, start_frame, end_frame):
    """Validate bake range against pin ranges."""
    all_pins_start_frame = set()
    all_pins_end_frame = set()
    for pin in pins_to_bake:
        pin_start = cmds.getAttr(pin + '.start_frame')
        pin_end = cmds.getAttr(pin + '.end_frame')
        all_pins_start_frame.add(pin_start)
        all_pins_end_frame.add(pin_end)
    all_pins_start_frame = list(all_pins_start_frame)[0]
    all_pins_end_frame = list(all_pins_end_frame)[0]

    selected_range = str(cmds.timeControl(
        aPlayBackSliderPython,
        q=True,
        range=True))
    selected_range = [float(x) for x in selected_range.strip('"').split(':')]
    if not selected_range[1] - 1 == selected_range[0]:
        start_frame, end_frame = selected_range

    start_frame, end_frame = validate_framerange(start_frame, end_frame)

    if start_frame < all_pins_start_frame:
        start_frame = all_pins_start_frame

    if end_frame > all_pins_end_frame:
        end_frame = all_pins_end_frame

    return start_frame, end_frame


def is_locked_or_not_keyable(controlFN, attribute):
    """Check if an attribute is locked or not keyable."""
    plug = controlFN.findPlug(attribute, False)
    if any(plug.child(p).isLocked for p in range(plug.numChildren())):
        return True
    if not any(plug.child(p).isKeyable for p in range(plug.numChildren())):
        return True
    for p in range(plug.numChildren()):
        plug_array = plug.child(p).connectedTo(True, False)
        if plug_array:
            if plug_array[0].isChild:
                # Check if the connected node is an animation blend node (from anim layers)
                # Animation layer connections should not be treated as "locked"
                conn_node = api.MFnDependencyNode(plug_array[0].node())
                conn_type = conn_node.typeName
                if 'animBlend' in conn_type:
                    continue  # Skip animation layer connections
                return True
    return False


# =============================================================================
# Key and Curve Functions
# =============================================================================

def get_keys_from_obj_attribute(controlFN, attribute):
    """Get keyframe times from an attribute."""
    keys = set()
    attribute_plug = controlFN.findPlug(attribute, False)
    if attribute_plug.isCompound:
        for c in range(attribute_plug.numChildren()):
            plug = attribute_plug.child(c)
            if anim.MAnimUtil.isAnimated(plug):
                keys.update(get_keys_from_curve(plug))
    else:
        if anim.MAnimUtil.isAnimated(attribute_plug):
            keys.update(get_keys_from_curve(attribute_plug))
    return list(keys)


def get_keys_from_curve(plug):
    """Get keyframe times from an animation curve."""
    curve = anim.MFnAnimCurve(plug)
    return [curve.input(k).value for k in range(curve.numKeys)]


def bookend_curves(controls, start_frame, end_frame):
    """
    Insert bookend keys one frame before/after the work area to preserve curve shape.
    """
    attrs = ['translateX', 'translateY', 'translateZ',
             'rotateX', 'rotateY', 'rotateZ']

    for control in controls:
        for attr in attrs:
            plug = f'{control}.{attr}'

            if not cmds.objExists(plug):
                continue
            if not cmds.getAttr(plug, keyable=True):
                continue

            key_times = cmds.keyframe(plug, query=True, timeChange=True) or []

            if not key_times:
                continue

            keys_before = [t for t in key_times if t < start_frame]
            if keys_before:
                bookend_time = start_frame - 1
                if bookend_time not in key_times:
                    cmds.setKeyframe(plug, time=bookend_time, insert=True)

            keys_after = [t for t in key_times if t > end_frame]
            if keys_after:
                bookend_time = end_frame + 1
                if bookend_time not in key_times:
                    cmds.setKeyframe(plug, time=bookend_time, insert=True)


def get_long_name(mobject):
    """Get the full DAG path (long name) for a Maya object.

    This ensures unique identification even when multiple objects
    share the same short name.
    """
    try:
        # Use MFnDagNode to get the full path directly
        dag_fn = api.MFnDagNode(mobject)
        return dag_fn.fullPathName()
    except (TypeError, RuntimeError):
        # Fall back to dependency node name if not a DAG node
        fn = api.MFnDependencyNode(mobject)
        return fn.name()


def get_short_name(long_name):
    """Extract the short name from a long DAG path.

    Example: '|group1|arm_ctrl' -> 'arm_ctrl'
    Also handles namespaces: '|group1|ns:arm_ctrl' -> 'arm_ctrl'
    """
    short = long_name.split('|')[-1]  # Get last part of path
    short = short.split(':')[-1]  # Remove namespace
    return short


def read_control_data(control, start_frame, end_frame):
    """Read control data for pin creation."""
    control_data = OrderedDict()
    controlFN = api.MFnDependencyNode(control)
    # Use long name (full DAG path) to uniquely identify the control
    control_name = get_long_name(control)

    t_keys = get_keys_from_obj_attribute(controlFN, 'translate')
    r_keys = get_keys_from_obj_attribute(controlFN, 'rotate')

    t_lock = is_locked_or_not_keyable(controlFN, 'translate')
    r_lock = is_locked_or_not_keyable(controlFN, 'rotate')

    bp_keys = []
    if 'blendParent1' in cmds.listAttr(control_name):
        bp_key_times = get_keys_from_obj_attribute(controlFN, 'blendParent1')
        if bp_key_times:
            key_values = []
            for time in bp_key_times:
                key_value = cmds.getAttr(control_name + '.blendParent1', time=time)
                key_values.append(key_value)
            bp_keys = list(zip(bp_key_times, key_values))
        else:
            bp_keys = [cmds.getAttr(control_name + '.blendParent1')]

    control_data['control'] = control_name
    control_data['start_frame'] = start_frame
    control_data['end_frame'] = end_frame
    control_data['translate_keys'] = t_keys
    control_data['rotate_keys'] = r_keys
    control_data['translate_locked'] = t_lock
    control_data['rotate_locked'] = r_lock
    control_data['preserve_blendParent'] = bp_keys

    return control_data


# =============================================================================
# Baking Functions
# =============================================================================

@viewport_off
def do_bake(nodes_to_bake, start_frame, end_frame, sample=1, destinationLayer=None):
    """Bake animation to nodes."""
    try:
        bake_kwargs = {
            'simulation': True,
            'time': (start_frame, end_frame),
            'sampleBy': sample,
            'oversamplingRate': 1,
            'disableImplicitControl': True,
            'preserveOutsideKeys': True,
            'sparseAnimCurveBake': False,
            'removeBakedAttributeFromLayer': False,
            'removeBakedAnimFromLayer': False,
            'bakeOnOverrideLayer': False,
            'minimizeRotation': True,
            'controlPoints': False,
            'shape': True
        }

        if destinationLayer:
            bake_kwargs['at'] = ("tx", "ty", "tz", "rx", "ry", "rz")
            bake_kwargs['destinationLayer'] = destinationLayer
        else:
            bake_kwargs['at'] = ("tx", "ty", "tz", "rx", "ry", "rz", "blendParent1")

        cmds.bakeResults(nodes_to_bake, **bake_kwargs)
        return True
    except:
        return False


@viewport_off
def do_bake_to_layer(controls_to_bake, start_frame, end_frame, sample=1,
                     unrollRotations=True, debug=False, constraints=None):
    """
    Hybrid bake approach: Bake to override layer, then convert to additive.
    This is ~3x faster than frame-by-frame Python baking for animation layers.
    """
    original_time = cmds.currentTime(query=True)

    try:
        layer_name = 'AnimPin_Layer'
        counter = 1
        while cmds.objExists(layer_name):
            layer_name = f'AnimPin_Layer_{counter}'
            counter += 1

        layer = cmds.animLayer(layer_name, override=True)

        bake_kwargs = {
            'simulation': True,
            'time': (start_frame, end_frame),
            'sampleBy': sample,
            'oversamplingRate': 1,
            'disableImplicitControl': True,
            'preserveOutsideKeys': True,
            'sparseAnimCurveBake': False,
            'minimizeRotation': True,
            'destinationLayer': layer,
            'at': ("tx", "ty", "tz", "rx", "ry", "rz")
        }
        cmds.bakeResults(controls_to_bake, **bake_kwargs)

        if constraints:
            for constraint in constraints:
                if cmds.objExists(constraint):
                    cmds.delete(constraint)

        convertOverrideToAdditive(layer)

        if unrollRotations:
            eulerFilterLayer(layer, nodes=controls_to_bake)

        return True

    except Exception as e:
        api.MGlobal.displayError(f"Bake to layer failed: {e}")
        return False

    finally:
        cmds.currentTime(original_time, edit=True)


def match_keys_procedure(pins_to_bake, start_frame, end_frame, composite=True):
    """Match keys procedure after baking."""
    for pin in pins_to_bake:
        control = cmds.getAttr(pin + '.control')

        translate_keys = cmds.getAttr(pin + '.translate_keys') or []
        if translate_keys:
            translate_keys = [float(x) for x in translate_keys.split(' ')]
            float_translate_keys = translate_keys[:]
            for key in float_translate_keys:
                if not key.is_integer():
                    translate_keys.remove(key)
                    translate_keys.append(int(round(key)))

        translate_keys_baked = set(cmds.keyframe(
            control,
            attribute='t',
            time=(min(translate_keys or [start_frame]), max(translate_keys or [end_frame])),
            query=True) or [])
        translate_keys_to_remove = list(set(translate_keys_baked - set(translate_keys)))

        rotate_keys = cmds.getAttr(pin + '.rotate_keys') or []
        if rotate_keys:
            rotate_keys = [float(x) for x in rotate_keys.split(' ')]
            float_rotate_keys = rotate_keys[:]
            for key in float_rotate_keys:
                if not key.is_integer():
                    rotate_keys.remove(key)
                    rotate_keys.append(int(round(key)))

        rotate_keys_baked = set(cmds.keyframe(
            control,
            attribute='r',
            time=(min(rotate_keys or [start_frame]), max(rotate_keys or [end_frame])),
            query=True) or [])
        rotate_keys_to_remove = list(set(rotate_keys_baked - set(rotate_keys)))

        keys_baked = list(translate_keys_baked | rotate_keys_baked)
        if composite:
            composited_keys = list(set(translate_keys + rotate_keys))
            keys_to_remove = list(set(keys_baked) - set(composited_keys))
            for key in keys_to_remove:
                cmds.cutKey(control, t=(key,), attribute=('t', 'r'), clear=True)

        keys_baked.insert(0, keys_baked[0] - 1)
        keys_baked.append(keys_baked[-1] + 1)
        keys = []
        bp_keys = cmds.getAttr(pin + '.preserve_blendParent')
        for match in tuple_rex.finditer(bp_keys):
            keys.append(float(match.group(0).split(', ')[0]))
        bp_keys_to_remove = list(set(keys_baked) - set(keys))
        for key in _to_ranges(bp_keys_to_remove):
            cmds.cutKey(control, time=key, attribute=('blendParent1'), clear=True)

    return True


def _to_ranges(iterable):
    """Convert list of integers to ranges."""
    iterable = sorted(set(iterable))
    for key, group in itertools.groupby(enumerate(iterable), lambda t: t[1] - t[0]):
        group = list(group)
        yield group[0][1], group[-1][1]


# =============================================================================
# Layer Detection Functions
# =============================================================================

def control_has_anim_layers(control):
    """Check if a control has any attributes on animation layers (excluding BaseAnimation).

    Uses long names (full DAG paths) for comparison to correctly handle
    multiple objects with the same short name.
    """
    all_layers = cmds.ls(type='animLayer') or []

    # Normalize control to long name for accurate comparison
    control_long_names = cmds.ls(control, long=True) or []
    if not control_long_names:
        return False
    control_long = control_long_names[0]

    for layer in all_layers:
        if layer == 'BaseAnimation':
            continue

        layer_plugs = cmds.animLayer(layer, query=True, layeredPlug=True) or []
        for plug in layer_plugs:
            plug_node = plug.split('.')[0]
            # Convert to long name for comparison
            plug_node_long_names = cmds.ls(plug_node, long=True) or []
            if plug_node_long_names and plug_node_long_names[0] == control_long:
                return True

        layer_curves = cmds.animLayer(layer, query=True, animCurves=True) or []
        for curve in layer_curves:
            connections = cmds.listConnections(curve + '.output', plugs=True) or []
            for conn in connections:
                conn_node = conn.split('.')[0]
                # Convert to long name for comparison
                conn_node_long_names = cmds.ls(conn_node, long=True) or []
                if conn_node_long_names and conn_node_long_names[0] == control_long:
                    return True

                node_type = cmds.nodeType(conn_node)
                if 'animBlend' in node_type:
                    blend_outputs = cmds.listConnections(conn_node + '.output', plugs=True) or []
                    for bout in blend_outputs:
                        bout_node = bout.split('.')[0]
                        # Convert to long name for comparison
                        bout_long_names = cmds.ls(bout_node, long=True) or []
                        if bout_long_names and bout_long_names[0] == control_long:
                            return True

    return False


def find_pin_for_control(control, all_pin_groups=None):
    """Find the pin locator and pin_group for a given control.

    Args:
        control: Control name (short or long) to find pin for
        all_pin_groups: Optional list of pin groups to search

    Returns:
        Tuple of (pin, pin_group) if found, or (None, None) if not found
    """
    if all_pin_groups is None:
        all_pin_groups = get_pin_groups()

    all_pins = get_pins(all_pin_groups)

    # Normalize input control name to long name for comparison
    control_long_names = cmds.ls(control, long=True) or []
    if not control_long_names:
        return (None, None)
    control_long_name = control_long_names[0]

    for pin in all_pins:
        pin_control = cmds.getAttr(pin + '.control')
        # Compare using long names for unique identification
        if pin_control == control_long_name:
            parent = cmds.listRelatives(pin, parent=True)
            while parent:
                parent = parent[0]
                if parent in all_pin_groups:
                    return (pin, parent)
                parent = cmds.listRelatives(parent, parent=True)

    return (None, None)


def find_pin_groups_from_selection(selection):
    """Find pin groups from selection."""
    all_pin_groups = get_pin_groups()
    pin_group_list = set()

    for sel in selection:
        if sel in all_pin_groups:
            pin_group_list.add(sel)
        elif cmds.attributeQuery('control', node=sel, exists=True):
            parent = cmds.listRelatives(sel, parent=True)
            while parent:
                parent = parent[0]
                if parent in all_pin_groups:
                    pin_group_list.add(parent)
                    break
                parent = cmds.listRelatives(parent, parent=True)
        else:
            pin, pin_group = find_pin_for_control(sel, all_pin_groups)
            if pin_group:
                pin_group_list.add(pin_group)
                continue

            for pg in all_pin_groups:
                constraints = cmds.listRelatives(pg, type='parentConstraint') or []
                for constraint in constraints:
                    targets = cmds.parentConstraint(constraint, query=True, targetList=True) or []
                    if sel in targets:
                        pin_group_list.add(pg)
                        break

    return pin_group_list


# =============================================================================
# Constraint Functions
# =============================================================================

def parent_constraint_with_skips(driver, driven):
    """Create a parent constraint from the driver to the driven, skipping locked axes."""
    skipT = []
    skipR = []

    for axis in ['X', 'Y', 'Z']:
        t_attr = f"{driven}.translate{axis}"
        r_attr = f"{driven}.rotate{axis}"

        if cmds.getAttr(t_attr, lock=True):
            skipT.append(axis.lower())
        if cmds.getAttr(r_attr, lock=True):
            skipR.append(axis.lower())

    constraint = ""
    try:
        constraint = cmds.parentConstraint(
            driver,
            driven,
            mo=False,
            skipTranslate=skipT,
            skipRotate=skipR
        )[0] or ""
    except Exception as e:
        log.warning(f"Could not apply parentConstraint: {e}")

    return constraint


# =============================================================================
# Layer Conversion Main Function
# =============================================================================

def convertOverrideToAdditive(layer, debug=False, verify=False, tolerance=0.01):
    """
    Convert an override animation layer to an additive layer.

    This modifies the layer's animation curves in place, replacing absolute
    values with delta values relative to the layer stack below.
    """
    if not cmds.objExists(layer):
        raise ValueError(f"Layer '{layer}' does not exist")

    if cmds.nodeType(layer) != 'animLayer':
        raise ValueError(f"'{layer}' is not an animation layer")

    is_valid, error_msg = validateLayerState(layer)
    if not is_valid:
        raise ValueError(error_msg)

    is_override = cmds.getAttr(f"{layer}.override")
    if not is_override:
        log.warning(f"Layer '{layer}' is already additive, nothing to convert")
        return {'curves_converted': 0, 'keys_processed': 0, 'nodes_affected': []}

    layer_attrs = cmds.animLayer(layer, query=True, attribute=True) or []
    nodes = list(set(a.split('.')[0] for a in layer_attrs if '.' in a))

    if not nodes:
        log.warning(f"No nodes found on layer '{layer}'")
        return {'curves_converted': 0, 'keys_processed': 0, 'nodes_affected': []}

    stats = {
        'curves_converted': 0,
        'keys_processed': 0,
        'nodes_affected': [],
    }

    cmds.undoInfo(openChunk=True, chunkName=f"Convert {layer} to Additive")
    try:
        for node in nodes:
            try:
                stack = LayerStack.build(node, layer)

                if not stack.target_curves:
                    continue

                layer_info = stack.layers[layer]
                rotation_mode = layer_info.rotation_mode

                rot_attrs = {'rotateX', 'rotateY', 'rotateZ'}
                has_all_rotations = rot_attrs.issubset(set(stack.target_curves.keys()))

                if rotation_mode == 1 and has_all_rotations:
                    keys_converted = _convertRotationsWithStack(node, stack)
                    stats['curves_converted'] += 3
                    stats['keys_processed'] += keys_converted
                    processed_attrs = rot_attrs
                else:
                    processed_attrs = set()

                for attr, curve_fn in stack.target_curves.items():
                    if attr in processed_attrs:
                        continue

                    keys_converted = _convertCurveWithStack(attr, curve_fn, stack)
                    stats['curves_converted'] += 1
                    stats['keys_processed'] += keys_converted

                stats['nodes_affected'].append(node)

            except Exception as e:
                log.error(f"Error converting curves for '{node}': {e}")

        cmds.setAttr(f"{layer}.override", 0)

    finally:
        cmds.undoInfo(closeChunk=True)

    return stats


def validateLayerState(layer):
    """Validate that the layer can be converted."""
    if not cmds.objExists(layer):
        return False, f"Layer '{layer}' does not exist"

    if cmds.getAttr(f"{layer}.mute"):
        return False, f"Layer '{layer}' is muted. Unmute it before conversion."

    all_layers = cmds.ls(type='animLayer')
    for other_layer in all_layers:
        if other_layer != layer and cmds.objExists(f"{other_layer}.solo"):
            if cmds.getAttr(f"{other_layer}.solo"):
                if not cmds.getAttr(f"{layer}.solo"):
                    return False, f"Layer '{layer}' is muted because '{other_layer}' is soloed."

    return True, None


def _convertCurveWithStack(attr: str, curve_fn: anim.MFnAnimCurve, stack: 'LayerStack') -> int:
    """Convert a single curve using LayerStack for base values."""
    num_keys = curve_fn.numKeys
    if num_keys == 0:
        return 0

    curve_name = api.MFnDependencyNode(curve_fn.object()).name()

    key_times = []
    for i in range(num_keys):
        mtime = curve_fn.input(i)
        key_times.append(mtime.asUnits(api.MTime.uiUnit()))

    base_values = stack.get_base_values(attr, key_times)

    attr_long = normalizeAttrName(attr)
    is_rotation = attr_long in ROTATION_ATTRS
    is_scale = attr_long in SCALE_ATTRS

    layer_info = stack.layers[stack.target_layer]
    scale_mode = layer_info.scale_mode

    for i in range(num_keys):
        override_value = curve_fn.value(i)
        base_value = base_values[i]

        if is_rotation:
            override_deg = math.degrees(override_value)
            delta_deg = override_deg - base_value
            delta_value = delta_deg
        elif is_scale and scale_mode == 1:
            if abs(base_value) > 0.0001:
                delta_value = override_value / base_value
            else:
                delta_value = 1.0
        else:
            delta_value = override_value - base_value

        cmds.keyframe(curve_name, index=(i,), valueChange=delta_value, absolute=True)

    return num_keys


def _convertRotationsWithStack(node: str, stack: 'LayerStack') -> int:
    """Convert rotation curves using quaternion delta calculation."""
    curve_fns = {
        'rotateX': stack.target_curves['rotateX'],
        'rotateY': stack.target_curves['rotateY'],
        'rotateZ': stack.target_curves['rotateZ'],
    }

    curve_names = {
        attr: api.MFnDependencyNode(fn.object()).name()
        for attr, fn in curve_fns.items()
    }

    curve_x = curve_fns['rotateX']
    num_keys = curve_x.numKeys
    if num_keys == 0:
        return 0

    key_times = []
    for i in range(num_keys):
        mtime = curve_x.input(i)
        key_times.append(mtime.asUnits(api.MTime.uiUnit()))

    base_rx = stack.get_base_values('rotateX', key_times)
    base_ry = stack.get_base_values('rotateY', key_times)
    base_rz = stack.get_base_values('rotateZ', key_times)

    override_rx = [math.degrees(curve_fns['rotateX'].value(i)) for i in range(num_keys)]
    override_ry = [math.degrees(curve_fns['rotateY'].value(i)) for i in range(num_keys)]
    override_rz = [math.degrees(curve_fns['rotateZ'].value(i)) for i in range(num_keys)]

    delta_rx, delta_ry, delta_rz = computeQuaternionDelta(
        override_rx, override_ry, override_rz,
        base_rx, base_ry, base_rz
    )

    for i in range(num_keys):
        cmds.keyframe(curve_names['rotateX'], index=(i,), valueChange=delta_rx[i], absolute=True)
        cmds.keyframe(curve_names['rotateY'], index=(i,), valueChange=delta_ry[i], absolute=True)
        cmds.keyframe(curve_names['rotateZ'], index=(i,), valueChange=delta_rz[i], absolute=True)

    return num_keys * 3


# =============================================================================
# Quaternion Math
# =============================================================================

def computeQuaternionDelta(override_rx, override_ry, override_rz,
                           base_rx, base_ry, base_rz):
    """Compute rotation delta using quaternion math: delta = override * inverse(base)"""
    delta_rx = []
    delta_ry = []
    delta_rz = []

    for i in range(len(override_rx)):
        o_rx = math.radians(override_rx[i])
        o_ry = math.radians(override_ry[i])
        o_rz = math.radians(override_rz[i])
        b_rx = math.radians(base_rx[i])
        b_ry = math.radians(base_ry[i])
        b_rz = math.radians(base_rz[i])

        override_quat = eulerToQuaternion(o_rx, o_ry, o_rz)
        base_quat = eulerToQuaternion(b_rx, b_ry, b_rz)

        base_inv = quaternionInverse(base_quat)
        delta_quat = quaternionMultiply(override_quat, base_inv)

        d_rx, d_ry, d_rz = quaternionToEuler(delta_quat)

        delta_rx.append(math.degrees(d_rx))
        delta_ry.append(math.degrees(d_ry))
        delta_rz.append(math.degrees(d_rz))

    return delta_rx, delta_ry, delta_rz


def eulerToQuaternion(rx, ry, rz):
    """Convert XYZ euler angles (radians) to quaternion [w, x, y, z]."""
    cx, sx = math.cos(rx / 2), math.sin(rx / 2)
    cy, sy = math.cos(ry / 2), math.sin(ry / 2)
    cz, sz = math.cos(rz / 2), math.sin(rz / 2)

    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz

    return [w, x, y, z]


def quaternionInverse(q):
    """Compute inverse of a unit quaternion."""
    return [q[0], -q[1], -q[2], -q[3]]


def quaternionMultiply(q1, q2):
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return [w, x, y, z]


def quaternionPower(q, t):
    """
    Raise a quaternion to a power t.

    For unit quaternions, q^t interpolates between identity (t=0) and q (t=1).
    Used to "un-slerp" weighted rotations: if S = slerp(identity, D, w),
    then D = S^(1/w).

    Args:
        q: Quaternion as [w, x, y, z]
        t: Power to raise to

    Returns:
        Quaternion [w, x, y, z] representing q^t
    """
    w, x, y, z = q

    # Handle identity quaternion (no rotation)
    vec_len = math.sqrt(x*x + y*y + z*z)
    if vec_len < 1e-10:
        return [1.0, 0.0, 0.0, 0.0]

    # Convert to angle-axis representation
    # q = cos(θ/2) + sin(θ/2) * axis
    # w = cos(θ/2), vec_len = sin(θ/2)
    half_angle = math.atan2(vec_len, w)

    # Scale the angle by power t
    new_half_angle = half_angle * t

    # Convert back to quaternion
    axis_x = x / vec_len
    axis_y = y / vec_len
    axis_z = z / vec_len

    new_w = math.cos(new_half_angle)
    new_sin = math.sin(new_half_angle)
    new_x = axis_x * new_sin
    new_y = axis_y * new_sin
    new_z = axis_z * new_sin

    return [new_w, new_x, new_y, new_z]


def quaternionToEuler(q):
    """Convert quaternion to XYZ euler angles (radians)."""
    w, x, y, z = q

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    rx = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        ry = math.copysign(math.pi / 2, sinp)
    else:
        ry = math.asin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    rz = math.atan2(siny_cosp, cosy_cosp)

    return rx, ry, rz


# =============================================================================
# Euler Filter
# =============================================================================

def eulerFilter(rot_x, rot_y, rot_z):
    """
    Euler filter using NumPy vectorization.
    Handles gimbal lock and rotation wrapping to prevent flips.
    """
    if len(rot_x) < 2:
        return rot_x, rot_y, rot_z

    rx = np.array(rot_x, dtype=np.float64)
    ry = np.array(rot_y, dtype=np.float64)
    rz = np.array(rot_z, dtype=np.float64)

    n = len(rx)

    filtered_x = np.empty(n, dtype=np.float64)
    filtered_y = np.empty(n, dtype=np.float64)
    filtered_z = np.empty(n, dtype=np.float64)

    filtered_x[0] = rx[0]
    filtered_y[0] = ry[0]
    filtered_z[0] = rz[0]

    fine_offsets = np.array([0, 360, -360], dtype=np.float64)
    ox, oy, oz = np.meshgrid(fine_offsets, fine_offsets, fine_offsets, indexing='ij')
    offset_combos = np.stack([ox.ravel(), oy.ravel(), oz.ravel()], axis=1)

    for i in range(1, n):
        prev = np.array([filtered_x[i - 1], filtered_y[i - 1], filtered_z[i - 1]])
        current = np.array([rx[i], ry[i], rz[i]])

        delta = current - prev
        n_rotations = np.round(delta / 360.0) * 360.0
        unwrapped = current - n_rotations

        candidates = unwrapped + offset_combos
        distances = np.abs(candidates - prev).sum(axis=1)
        best_idx = np.argmin(distances)
        best = candidates[best_idx]
        best_dist = distances[best_idx]

        if best_dist > 90:
            gimbal_variants = np.array([
                [current[0] + 180, 180 - current[1], current[2] + 180],
                [current[0] - 180, 180 - current[1], current[2] - 180],
                [current[0] + 180, -180 - current[1], current[2] + 180],
                [current[0] - 180, -180 - current[1], current[2] - 180],
            ], dtype=np.float64)

            for gv in gimbal_variants:
                g_delta = gv - prev
                g_n_rot = np.round(g_delta / 360.0) * 360.0
                g_unwrapped = gv - g_n_rot

                g_candidates = g_unwrapped + offset_combos
                g_distances = np.abs(g_candidates - prev).sum(axis=1)
                g_best_idx = np.argmin(g_distances)
                g_best_dist = g_distances[g_best_idx]

                if g_best_dist < best_dist:
                    best_dist = g_best_dist
                    best = g_candidates[g_best_idx]

        filtered_x[i] = best[0]
        filtered_y[i] = best[1]
        filtered_z[i] = best[2]

    return filtered_x.tolist(), filtered_y.tolist(), filtered_z.tolist()


def eulerFilterLayer(layer, nodes=None):
    """Apply euler filter to rotation curves on an animation layer."""
    if not cmds.objExists(layer):
        raise RuntimeError(f"Layer '{layer}' does not exist")

    layer_curves = getLayerAnimCurves(layer)

    if not layer_curves:
        return {}

    node_curves = {}
    for curve_node, (node, attr) in layer_curves.items():
        if nodes is not None:
            if isinstance(nodes, str):
                nodes = [nodes]
            if node not in nodes:
                continue

        if node not in node_curves:
            node_curves[node] = {}
        node_curves[node][attr] = curve_node

    processed = {}

    for node, curves in node_curves.items():
        rot_attrs = ['rotateX', 'rotateY', 'rotateZ']
        if not all(attr in curves for attr in rot_attrs):
            continue

        curve_fns = {}
        for attr in rot_attrs:
            curve_name = curves[attr]
            sel_list = api.MSelectionList()
            sel_list.add(curve_name)
            curve_fns[attr] = anim.MFnAnimCurve(sel_list.getDependNode(0))

        curve_x = curve_fns['rotateX']
        curve_y = curve_fns['rotateY']
        curve_z = curve_fns['rotateZ']

        num_keys = curve_x.numKeys
        if num_keys < 2:
            continue

        rot_x = []
        rot_y = []
        rot_z = []

        for i in range(num_keys):
            rot_x.append(math.degrees(curve_x.value(i)))
            rot_y.append(math.degrees(curve_y.value(i)))
            rot_z.append(math.degrees(curve_z.value(i)))

        filtered_x, filtered_y, filtered_z = eulerFilter(rot_x, rot_y, rot_z)

        changes_made = False
        for i in range(num_keys):
            if (abs(filtered_x[i] - rot_x[i]) > 0.001 or
                abs(filtered_y[i] - rot_y[i]) > 0.001 or
                abs(filtered_z[i] - rot_z[i]) > 0.001):
                changes_made = True
                break

        if not changes_made:
            continue

        for i in range(num_keys):
            curve_x.setValue(i, math.radians(filtered_x[i]))
            curve_y.setValue(i, math.radians(filtered_y[i]))
            curve_z.setValue(i, math.radians(filtered_z[i]))

        processed[node] = num_keys

    return processed


def getLayerAnimCurves(layer):
    """Get all animation curves on a layer and their associated node.attr pairs."""
    curves = {}
    layer_curves = cmds.animLayer(layer, query=True, animCurves=True) or []

    for curve in layer_curves:
        node_attr = getCurveTarget(curve)
        if node_attr:
            node, attr = node_attr
            curves[curve] = (node, attr)

    return curves


def getCurveTarget(curve):
    """Find the node and attribute that an animation curve controls."""
    if not cmds.objExists(curve):
        return None

    node_type = cmds.nodeType(curve)
    if not node_type.startswith('animCurve'):
        return None

    outputs = cmds.listConnections(f"{curve}.output", source=False,
                                    destination=True, plugs=True) or []

    for output_plug in outputs:
        if '.input' in output_plug.lower():
            target = traceBlendToTarget(output_plug)
            if target:
                return target
        elif '.' in output_plug:
            parts = output_plug.rsplit('.', 1)
            if len(parts) == 2:
                return (parts[0], parts[1])

    return None


def traceBlendToTarget(blend_plug):
    """Trace from a blend node input to the final target attribute."""
    if '.' not in blend_plug:
        return None

    blend_node, input_attr = blend_plug.split('.', 1)

    output_attr = "output"
    if input_attr.endswith('X'):
        output_attr = "outputX"
    elif input_attr.endswith('Y'):
        output_attr = "outputY"
    elif input_attr.endswith('Z'):
        output_attr = "outputZ"

    visited = set()
    current = blend_node
    current_output = output_attr

    while current and current not in visited:
        visited.add(current)

        full_output = f"{current}.{current_output}"
        outputs = []

        if cmds.objExists(full_output):
            outputs = cmds.listConnections(full_output, source=False,
                                            destination=True, plugs=True) or []

        if not outputs and current_output != "output":
            generic_output = f"{current}.output"
            if cmds.objExists(generic_output):
                outputs = cmds.listConnections(generic_output, source=False,
                                                destination=True, plugs=True) or []

        if not outputs:
            break

        for output_plug in outputs:
            if '.' not in output_plug:
                continue

            target_node, target_attr = output_plug.split('.', 1)
            target_type = cmds.nodeType(target_node)

            if 'Blend' in target_type or 'blend' in target_type or target_type == 'pairBlend':
                current = target_node
                if target_type == 'pairBlend':
                    if 'Translate' in target_attr:
                        axis = target_attr[-2] if target_attr[-1].isdigit() else target_attr[-1]
                        current_output = f"outTranslate{axis.upper()}"
                    elif 'Rotate' in target_attr:
                        axis = target_attr[-2] if target_attr[-1].isdigit() else target_attr[-1]
                        current_output = f"outRotate{axis.upper()}"
                break
            else:
                parts = output_plug.rsplit('.', 1)
                if len(parts) == 2:
                    return (parts[0], parts[1])
        else:
            break

    return None


def unrollAdditiveRotations(layer, nodes=None, debug=False):
    """
    Unroll rotation curves on an additive animation layer.

    This function accounts for the layer stack (composite base values),
    the layer's rotation accumulation mode, and animated layer weights.

    For additive layers, the "correct" Euler representation of delta values
    depends on context. This function:
    1. Computes the composed rotation result at each keyframe
    2. Euler filters the composed result to remove discontinuities
    3. Derives new delta values that produce the filtered composed result

    For component mode (rotationAccumulationMode=0):
        composed = base + (delta * weight)
        new_delta = (filtered_composed - base) / weight

    For quaternion mode (rotationAccumulationMode=1):
        composed = base_quat * slerp(identity, delta_quat, weight)
        new_delta = (inv(base) * filtered)^(1/weight)

    Args:
        layer: Name of the additive animation layer
        nodes: Optional list of nodes to process. If None, processes all nodes
               on the layer. Nodes must have all three rotation channels
               (rotateX, rotateY, rotateZ) on the layer to be processed.
        debug: If True, print diagnostic information about the filtering process.

    Returns:
        dict: {node: num_keys_processed} for each processed node

    Raises:
        ValueError: If layer doesn't exist or is in override mode
    """
    if not cmds.objExists(layer):
        raise ValueError(f"Layer '{layer}' does not exist")

    if cmds.nodeType(layer) != 'animLayer':
        raise ValueError(f"'{layer}' is not an animation layer")

    # Check if layer is additive (override = 0 means additive)
    is_override = cmds.getAttr(f"{layer}.override")
    if is_override:
        log.warning(f"Layer '{layer}' is in override mode. Use eulerFilterLayer() instead.")
        return {}

    # Get all nodes on the layer
    layer_attrs = cmds.animLayer(layer, query=True, attribute=True) or []
    layer_nodes = set(a.split('.')[0] for a in layer_attrs if '.' in a)

    if not layer_nodes:
        log.warning(f"No nodes found on layer '{layer}'")
        return {}

    # Filter to requested nodes if provided
    if nodes is not None:
        if isinstance(nodes, str):
            nodes = [nodes]
        # Only process nodes that are both requested AND on the layer
        nodes_to_process = [n for n in nodes if n in layer_nodes]
        if not nodes_to_process:
            log.warning(f"None of the specified nodes are on layer '{layer}'")
            return {}
    else:
        nodes_to_process = list(layer_nodes)

    processed = {}

    for node in nodes_to_process:
        try:
            # Build layer stack for this node
            stack = LayerStack.build(node, layer)

            # Check if we have all rotation curves - euler filter requires all 3 axes
            rot_attrs = ['rotateX', 'rotateY', 'rotateZ']
            missing_attrs = [attr for attr in rot_attrs if attr not in stack.target_curves]
            if missing_attrs:
                log.warning(
                    f"Skipping '{node}': missing rotation channels {missing_attrs} on layer. "
                    f"Euler filtering requires all 3 rotation axes."
                )
                continue

            curve_fns = {attr: stack.target_curves[attr] for attr in rot_attrs}

            # Get curve names for writing back
            curve_names = {
                attr: api.MFnDependencyNode(fn.object()).name()
                for attr, fn in curve_fns.items()
            }

            # Collect all key times (union of all rotation curves)
            key_times_set = set()
            for attr, curve_fn in curve_fns.items():
                for i in range(curve_fn.numKeys):
                    mtime = curve_fn.input(i)
                    key_times_set.add(mtime.asUnits(api.MTime.uiUnit()))

            key_times = sorted(key_times_set)

            if len(key_times) < 2:
                continue

            # Get rotation accumulation mode
            layer_info = stack.layers[layer]
            rotation_mode = layer_info.rotation_mode  # 0=component, 1=quaternion

            # Sample values at each key time
            delta_rx, delta_ry, delta_rz = [], [], []
            base_rx, base_ry, base_rz = [], [], []
            weights = []

            for t in key_times:
                mtime = api.MTime(t, api.MTime.uiUnit())

                # Get delta values from curves (convert from radians to degrees)
                delta_rx.append(math.degrees(curve_fns['rotateX'].evaluate(mtime)))
                delta_ry.append(math.degrees(curve_fns['rotateY'].evaluate(mtime)))
                delta_rz.append(math.degrees(curve_fns['rotateZ'].evaluate(mtime)))

                # Get base composite values
                base_rx.append(stack.get_base_value('rotateX', t))
                base_ry.append(stack.get_base_value('rotateY', t))
                base_rz.append(stack.get_base_value('rotateZ', t))

                # Get layer weight at this time
                weights.append(stack.get_layer_weight(layer, t))

            # Compute composed rotations
            composed_rx, composed_ry, composed_rz = [], [], []

            for i in range(len(key_times)):
                weight = weights[i]

                if rotation_mode == 0:
                    # Component mode: composed = base + (delta * weight)
                    composed_rx.append(base_rx[i] + delta_rx[i] * weight)
                    composed_ry.append(base_ry[i] + delta_ry[i] * weight)
                    composed_rz.append(base_rz[i] + delta_rz[i] * weight)
                else:
                    # Quaternion mode: composed = base_quat * slerp(identity, delta_quat, weight)
                    base_quat = eulerToQuaternion(
                        math.radians(base_rx[i]),
                        math.radians(base_ry[i]),
                        math.radians(base_rz[i])
                    )
                    delta_quat = eulerToQuaternion(
                        math.radians(delta_rx[i]),
                        math.radians(delta_ry[i]),
                        math.radians(delta_rz[i])
                    )

                    # slerp(identity, delta, weight) = delta^weight
                    weighted_delta = quaternionPower(delta_quat, weight)

                    # composed = base * weighted_delta
                    composed_quat = quaternionMultiply(base_quat, weighted_delta)

                    # Convert back to Euler
                    c_rx, c_ry, c_rz = quaternionToEuler(composed_quat)
                    composed_rx.append(math.degrees(c_rx))
                    composed_ry.append(math.degrees(c_ry))
                    composed_rz.append(math.degrees(c_rz))

            # Euler filter the composed rotations
            filtered_rx, filtered_ry, filtered_rz = eulerFilter(
                composed_rx, composed_ry, composed_rz
            )

            if debug:
                print(f"\n=== Debug for {node} ===")
                print(f"Rotation mode: {'component' if rotation_mode == 0 else 'quaternion'}")
                print(f"Num keys: {len(key_times)}")

                # Find indices where filtering made a difference
                changes = []
                for i in range(len(key_times)):
                    diff_rx = abs(filtered_rx[i] - composed_rx[i])
                    diff_ry = abs(filtered_ry[i] - composed_ry[i])
                    diff_rz = abs(filtered_rz[i] - composed_rz[i])
                    if diff_rx > 0.01 or diff_ry > 0.01 or diff_rz > 0.01:
                        changes.append((i, key_times[i], diff_rx, diff_ry, diff_rz))

                if changes:
                    print(f"Euler filter made {len(changes)} changes:")
                    for idx, t, dx, dy, dz in changes[:5]:  # Show first 5
                        print(f"  Frame {t}: composed=({composed_rx[idx]:.2f}, {composed_ry[idx]:.2f}, {composed_rz[idx]:.2f}) -> filtered=({filtered_rx[idx]:.2f}, {filtered_ry[idx]:.2f}, {filtered_rz[idx]:.2f})")
                else:
                    print("Euler filter made NO changes to composed rotations!")
                    # Show sample of composed values to see if there are discontinuities
                    print("Sample composed values (looking for ~360 degree jumps):")
                    for i in range(min(10, len(key_times))):
                        print(f"  Frame {key_times[i]}: composed=({composed_rx[i]:.2f}, {composed_ry[i]:.2f}, {composed_rz[i]:.2f})")
                    if len(key_times) > 10:
                        print("  ...")

                print(f"\nSample base values:")
                for i in range(min(5, len(key_times))):
                    print(f"  Frame {key_times[i]}: base=({base_rx[i]:.2f}, {base_ry[i]:.2f}, {base_rz[i]:.2f})")

                print(f"\nSample delta values (from layer curves):")
                for i in range(min(5, len(key_times))):
                    print(f"  Frame {key_times[i]}: delta=({delta_rx[i]:.2f}, {delta_ry[i]:.2f}, {delta_rz[i]:.2f})")

            # Derive new deltas
            new_delta_rx, new_delta_ry, new_delta_rz = [], [], []

            for i in range(len(key_times)):
                weight = weights[i]

                if abs(weight) < 1e-10:
                    # Weight is zero, delta has no effect - keep original
                    new_delta_rx.append(delta_rx[i])
                    new_delta_ry.append(delta_ry[i])
                    new_delta_rz.append(delta_rz[i])
                elif rotation_mode == 0:
                    # Component mode: new_delta = (filtered_composed - base) / weight
                    new_delta_rx.append((filtered_rx[i] - base_rx[i]) / weight)
                    new_delta_ry.append((filtered_ry[i] - base_ry[i]) / weight)
                    new_delta_rz.append((filtered_rz[i] - base_rz[i]) / weight)
                else:
                    # Quaternion mode: new_delta = (inv(base) * filtered)^(1/weight)
                    base_quat = eulerToQuaternion(
                        math.radians(base_rx[i]),
                        math.radians(base_ry[i]),
                        math.radians(base_rz[i])
                    )
                    filtered_quat = eulerToQuaternion(
                        math.radians(filtered_rx[i]),
                        math.radians(filtered_ry[i]),
                        math.radians(filtered_rz[i])
                    )

                    # S = inv(base) * filtered (what the weighted delta should be)
                    base_inv = quaternionInverse(base_quat)
                    weighted_delta_needed = quaternionMultiply(base_inv, filtered_quat)

                    # new_delta = S^(1/weight) to un-weight it
                    new_delta_quat = quaternionPower(weighted_delta_needed, 1.0 / weight)

                    # Convert to Euler
                    nd_rx, nd_ry, nd_rz = quaternionToEuler(new_delta_quat)
                    new_delta_rx.append(math.degrees(nd_rx))
                    new_delta_ry.append(math.degrees(nd_ry))
                    new_delta_rz.append(math.degrees(nd_rz))

            # For quaternion mode, the quaternion-to-euler conversion gives "canonical"
            # values that may undo our unwrapping. Apply euler filter to the new deltas
            # to ensure they're continuous (the deltas are what get stored in curves).
            if rotation_mode == 1:
                new_delta_rx, new_delta_ry, new_delta_rz = eulerFilter(
                    new_delta_rx, new_delta_ry, new_delta_rz
                )

            if debug:
                # Show changes in delta values
                delta_changes = []
                for i in range(len(key_times)):
                    diff_rx = abs(new_delta_rx[i] - delta_rx[i])
                    diff_ry = abs(new_delta_ry[i] - delta_ry[i])
                    diff_rz = abs(new_delta_rz[i] - delta_rz[i])
                    if diff_rx > 0.01 or diff_ry > 0.01 or diff_rz > 0.01:
                        delta_changes.append((i, key_times[i]))

                print(f"\nDelta values changed at {len(delta_changes)} frames")
                if delta_changes:
                    for idx, t in delta_changes[:5]:
                        print(f"  Frame {t}: old=({delta_rx[idx]:.2f}, {delta_ry[idx]:.2f}, {delta_rz[idx]:.2f}) -> new=({new_delta_rx[idx]:.2f}, {new_delta_ry[idx]:.2f}, {new_delta_rz[idx]:.2f})")

            # Build time->value lookup for writing back
            # Round times to avoid float precision issues in dictionary lookup
            def round_time(t):
                return round(t, 6)

            time_to_value = {
                'rotateX': {round_time(t): v for t, v in zip(key_times, new_delta_rx)},
                'rotateY': {round_time(t): v for t, v in zip(key_times, new_delta_ry)},
                'rotateZ': {round_time(t): v for t, v in zip(key_times, new_delta_rz)},
            }

            # Write new deltas back to curves
            for attr, curve_fn in curve_fns.items():
                curve_name = curve_names[attr]
                value_lookup = time_to_value[attr]

                for i in range(curve_fn.numKeys):
                    mtime = curve_fn.input(i)
                    t = round_time(mtime.asUnits(api.MTime.uiUnit()))

                    if t in value_lookup:
                        cmds.keyframe(curve_name, index=(i,),
                                      valueChange=value_lookup[t], absolute=True)

            processed[node] = len(key_times)

        except Exception as e:
            log.error(f"Error unrolling rotations for '{node}': {e}")

    return processed


# =============================================================================
# Transform Recording (for debugging)
# =============================================================================

@viewport_off
def record_transforms(controls, start_frame, end_frame, sample=1):
    """Record world-space transforms for controls across frame range."""
    transforms = {}
    current_time = cmds.currentTime(query=True)

    for ctrl in controls:
        transforms[ctrl] = {}
        for frame in range(int(start_frame), int(end_frame) + 1, sample):
            cmds.currentTime(frame, edit=True)
            ws_pos = cmds.xform(ctrl, query=True, worldSpace=True, translation=True)
            ws_rot = cmds.xform(ctrl, query=True, worldSpace=True, rotation=True)
            transforms[ctrl][frame] = (ws_pos, ws_rot)

    cmds.currentTime(current_time, edit=True)
    return transforms


def get_xform_at_frame(ctrl, frame):
    """Get world-space translation and rotation at a specific frame."""
    cmds.currentTime(frame, edit=True)
    ws_pos = cmds.xform(ctrl, query=True, worldSpace=True, translation=True)
    ws_rot = cmds.xform(ctrl, query=True, worldSpace=True, rotation=True)
    return ws_pos, ws_rot
