"""
Writes keys onto anim curves in bulk, as one undo step.

``cmds.setKeyframe`` costs a command call per key. ``MFnAnimCurve.addKeys``
takes a whole curve's keys in one call and is 10 to 50 times faster, but two
things about it are not obvious (both checked in Maya 2025):

* It does not change a key that already exists at one of the times, so those
  keys are set with ``setValue`` and only the new times go to ``addKeys``.
  A key that already exists keeps its tangents, as it does with
  ``cmds.setKeyframe``.
* API edits are not on Maya's undo queue. They are recorded in an
  ``MAnimCurveChange``, which the ``adjustmentBlendKeys`` command (in
  ``plug-ins/``) takes over and replays for Edit > Undo and Redo.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

from maya import cmds
import maya.api.OpenMaya as om
import maya.api.OpenMayaAnim as oma

from .maya_layers import to_ui_units

COMMAND = "adjustmentBlendKeys"
_PLUGIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plug-ins", COMMAND + ".py")

# The change on its way to the command. Command flags would mean parsing every
# value back out of an argument list, which costs more than writing the keys.
_handoff: Optional[oma.MAnimCurveChange] = None


def write_keys(writes: Sequence[Tuple[str, Sequence[float], Sequence[float]]]) -> None:
    """Key every curve on its frames, as one undo step.

    Args:
        writes: ``(anim curve name, frames, values)`` triples, with one value
            per frame, in UI units (degrees for a rotation curve).

    Either every curve is written or, on an error, none is.
    """
    global _handoff
    if not writes:
        return
    if not hasattr(cmds, COMMAND):
        cmds.loadPlugin(_PLUGIN, quiet=True)

    change = oma.MAnimCurveChange()
    try:
        _write(writes, change)
        _handoff = change
        getattr(cmds, COMMAND)()
    except Exception:
        change.undoIt()
        raise
    finally:
        _handoff = None


def take_change() -> oma.MAnimCurveChange:
    """Hand the finished change to the ``adjustmentBlendKeys`` command."""
    global _handoff
    change, _handoff = _handoff, None
    if change is None:
        raise RuntimeError(
            f"{COMMAND} only records an edit made by adjustment_blend.maya_keys.write_keys(); "
            "it does nothing when called by hand."
        )
    return change


def _write(writes, change: oma.MAnimCurveChange) -> None:
    unit = om.MTime.uiUnit()
    tangent = oma.MFnAnimCurve.kTangentGlobal  # the user's default, as cmds.setKeyframe uses

    # Per set of frames: where each frame sits, its MTime, and the times that
    # still need a key by which frames are keyed already. Nearly every curve
    # has the same frames and the same few keys, so they share these.
    grids: Dict[Tuple[float, ...], tuple] = {}

    selection = om.MSelectionList()
    for curve, frames, values in writes:
        if len(values) != len(frames):
            raise ValueError(f"{curve}: {len(values)} values for {len(frames)} frames")
        grid = grids.get(tuple(frames))
        if grid is None:
            floats = [float(frame) for frame in frames]
            grid = grids[tuple(frames)] = (
                {frame: i for i, frame in enumerate(floats)},
                [om.MTime(frame, unit) for frame in floats],
                {},
            )
        position, mtimes, new_times = grid

        selection.clear()
        selection.add(curve)
        curve_fn = oma.MFnAnimCurve(selection.getDependNode(0))
        to_internal = 1.0 / to_ui_units(curve_fn.animCurveType)

        keyed: Dict[int, int] = {}  # position in frames -> index of the key that is there
        for index in range(curve_fn.numKeys):
            at = position.get(curve_fn.input(index).asUnits(unit))
            if at is not None:
                keyed[at] = index

        # Existing keys first: adding keys shifts the indices.
        for at, index in keyed.items():
            curve_fn.setValue(index, values[at] * to_internal, change)

        shape = tuple(sorted(keyed))
        times = new_times.get(shape)
        if times is None:
            times = new_times[shape] = om.MTimeArray(
                [mtime for i, mtime in enumerate(mtimes) if i not in keyed])
        if len(times):
            new_values = om.MDoubleArray(
                [values[i] * to_internal for i in range(len(frames)) if i not in keyed])
            curve_fn.addKeys(times, new_values, tangent, tangent, True, change)
