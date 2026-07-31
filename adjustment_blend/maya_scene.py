"""
Scene I/O and layer discovery for Adjustment Blend.

Everything that reads state out of the running Maya session lives here:
discovering which layer is the adjustment target and which layers sit below it,
listing animated attributes, and collecting adjustment keys.

Note on the UI dependency
-------------------------
:func:`discover_layers` reads the *visible* stack order straight from the Anim
Layer editor (``AnimLayerTabanimLayerEditor``). That is deliberate: Maya does
not expose a reliable, ordered view of the layer stack through the command API
(``animLayer -children`` mishandles nesting, ``ls -type animLayer`` returns
creation order), whereas the editor shows the exact order the animator is
working against. The tool only ever operates on anim layers, so in practice the
editor is always open.

Callers that need to run without the UI (headless, pipeline, tests) should not
remove this query — they should bypass it by passing ``adjustment_layer`` and
``layers_below`` straight to :func:`adjustment_blend.run`.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Set, Tuple

from maya import cmds

from . import core

log = logging.getLogger(__name__)

# Name of the Anim Layer editor treeView widget that holds the visible stack.
ANIM_LAYER_TREEVIEW = "AnimLayerTabanimLayerEditor"


def get_selected_animlayers() -> List[str]:
    """Return the names of the anim layers currently selected in the editor."""
    layers = []
    for each in cmds.ls(type="animLayer"):
        if cmds.animLayer(each, query=True, selected=True):
            layers.append(each)
    return layers


def discover_layers() -> Optional[Tuple[str, List[str]]]:
    """Resolve the adjustment layer and the (unlocked) layers below it.

    Reads the selection and the visible stack order from the Anim Layer editor
    (see the module docstring for why this uses the UI). The adjustment layer is
    the last selected layer; everything beneath it in the stack — minus any
    locked layers — becomes the set to composite motion from.

    Returns:
        ``(adjustment_layer, layers_below)``, or ``None`` if there is no anim
        layer root in the scene.
    """
    root_layer = cmds.animLayer(query=True, root=True)
    if not root_layer:
        return None

    # The editor's treeView reports layers in their true visible stack order.
    all_layers = cmds.treeView(ANIM_LAYER_TREEVIEW, q=True, children=True)

    selected_layers = get_selected_animlayers()

    layers_below = all_layers[:]  # Copy the list
    if len(selected_layers) == 0 or selected_layers[-1] == "BaseAnimation":
        # Nothing useful selected: treat the top layer as the adjustment layer.
        selected_layers = [layers_below.pop()]
    elif len(selected_layers) == 1:
        index = layers_below.index(selected_layers[-1])
        del layers_below[index:]
    elif len(selected_layers) > 1:
        layers_below = selected_layers[:-1]

    adjustment_layer = selected_layers[-1]

    if not isinstance(layers_below, list):
        layers_below = [layers_below]

    # Locked layers can't be authored against, so drop them.
    locked = [layer for layer in layers_below if cmds.animLayer(layer, q=True, lock=True)]
    for layer in locked:
        layers_below.remove(layer)

    log.debug(
        "Adjustment layer: %s | layers below: %s | locked (skipped): %s",
        adjustment_layer, layers_below, locked,
    )

    return adjustment_layer, layers_below


def get_animated_attributes(node: str) -> List[str]:
    """Return the sorted list of animated ``node.attribute`` plugs on ``node``.

    Uses the OpenMaya 1.0 ``MAnimUtil`` helpers, which resolve animated plugs
    through layer memberships in one call.
    """
    import maya.OpenMaya as om1
    import maya.OpenMayaAnim as oma1

    # Get a MDagPath for the given node name.
    sel_list = om1.MSelectionList()
    sel_list.add(node)
    dag_path = om1.MDagPath()
    sel_list.getDagPath(0, dag_path)

    # Find all the animated plugs.
    plug_array = om1.MPlugArray()
    oma1.MAnimUtil.findAnimatedPlugs(dag_path, plug_array)

    attribute_names = []
    for i in range(plug_array.length()):
        plug = om1.MPlug(plug_array[i])
        attribute_names.append(plug.name())

    return sorted(attribute_names)


def filter_layers_by_objects(layers: List[str], objects: List[str]) -> List[str]:
    """Remove layers that don't contain any of the target objects."""
    root_layer = cmds.animLayer(q=True, root=True)
    filtered = []

    for layer in layers:
        if layer == root_layer:
            filtered.append(layer)
            continue

        layer_members = cmds.animLayer(layer, q=True, attribute=True) or []
        layer_objects = set(x.split(".")[0] for x in layer_members)

        if bool(layer_objects & set(objects)):
            filtered.append(layer)
        else:
            log.debug("Skipping layer %s - no target objects", layer)

    return filtered


def collect_adjustment_keys(
    objects: List[str],
    adjustment_layer: str,
    adjustment_layer_members: List[str],
) -> Set[float]:
    """Collect all keyframe times from the adjustment layer for target objects."""
    adjustment_keys: Set[float] = set()

    for obj in objects:
        for attribute in get_animated_attributes(obj):
            _, attr = attribute.split(".")
            if attr not in core.ATTRIBUTES:
                continue
            if attribute not in adjustment_layer_members:
                continue

            curve = cmds.animLayer(adjustment_layer, q=True, findCurveForPlug=attribute)
            if curve:
                keyframes = cmds.keyframe(curve, q=True) or []
                adjustment_keys.update(keyframes)

    return adjustment_keys
