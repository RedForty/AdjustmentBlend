"""
Scene I/O and layer discovery for Adjustment Blend.

Everything that reads state out of the running Maya session lives here:
discovering which layer is the adjustment target and which layers sit below it,
and finding the adjustment layer's curves and their keys.

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
from collections import defaultdict
from typing import List, NamedTuple, Optional, Sequence, Tuple

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


class LayerCurve(NamedTuple):
    """One of the adjustment layer's own curves."""
    obj: str
    attr: str
    curve: str
    keys: List[float]      # its key times, in order


def layer_curves(layer: str, objects: Sequence[str]) -> List[LayerCurve]:
    """Return the layer's own curves for the transform channels of ``objects``.

    Ordered by object, as given, then by attribute name. A member attribute
    that has no curve on the layer yet is left out.
    """
    attrs_by_object = defaultdict(list)
    for plug in cmds.animLayer(layer, q=True, attribute=True) or []:
        obj, _, attr = plug.rpartition(".")
        if attr in core.ATTRIBUTES:
            attrs_by_object[obj].append(attr)

    curves = []
    for obj in dict.fromkeys(objects):
        for attr in sorted(attrs_by_object.get(obj, ())):
            curve = cmds.animLayer(layer, q=True, findCurveForPlug=f"{obj}.{attr}")
            if not curve:
                continue
            curve = curve[0] if isinstance(curve, list) else curve
            keys = cmds.keyframe(curve, q=True, timeChange=True) or []
            curves.append(LayerCurve(obj, attr, curve, sorted(set(keys))))
    return curves
