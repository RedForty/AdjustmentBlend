"""
Scene I/O and layer discovery for Adjustment Blend.

Everything that reads state out of the running Maya session lives here:
which layer is the adjustment target, and the adjustment layer's curves and
their keys.

None of it needs the Anim Layer editor to be open. The stack order comes from
``animLayer -children`` walked from the root layer, which is the order Maya
composites the layers in, nested layers included.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import List, NamedTuple, Optional, Sequence

from maya import cmds

from . import core

log = logging.getLogger(__name__)


def stack_order(parent: Optional[str] = None) -> List[str]:
    """Return every layer above the root, from the bottom of the stack to the top.

    A layer's children come right after it.
    """
    parent = parent or cmds.animLayer(query=True, root=True)
    if not parent:
        return []
    layers = []
    for child in cmds.animLayer(parent, query=True, children=True) or []:
        layers.append(child)
        layers.extend(stack_order(child))
    return layers


def discover_adjustment_layer() -> Optional[str]:
    """Return the layer to densify, or ``None`` if the scene has no layers.

    That is the selected layer. With several selected it is the topmost of
    them in the stack, and with none selected it is the top of the stack.
    """
    layers = stack_order()
    if not layers:
        return None
    selected = [layer for layer in layers if cmds.animLayer(layer, query=True, selected=True)]
    adjustment_layer = (selected or layers)[-1]
    log.debug("Adjustment layer: %s (selected: %s)", adjustment_layer, selected)
    return adjustment_layer


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
