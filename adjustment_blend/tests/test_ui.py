"""Group "ui": what the panel's button tells the artist.

The panel itself needs Maya's window system. These checks call the function behind its button,
which runs the blend in one undo chunk and returns the message it shows.
"""
import maya.cmds as cmds

from adjustment_blend import maya_keys, ui

from . import common, scenes
from .common import Failure


def run(report):
    def check(name, function):
        common.check(report, "ui, " + name, function)

    cmds.undoInfo(state=True)
    check("a blend that works says applied, and is one undo step", _applied)
    check("nothing to blend says so", _nothing)
    check("a blend that breaks says failed", _failed)


def _applied():
    scene = scenes.move_hold_move_hold()  # leaves the control and the layer selected
    result, message = ui._run_and_report("scalar", True)
    if result is not True or "applied" not in message:
        raise Failure("returned %r and said %r" % (result, message))
    for attr in scene.keys:
        common.expect_holds(scene.control, attr)
    cmds.undo()
    for attr in scene.keys:
        common.expect_sparse(scene, scene.control, attr)
    return message


def _nothing():
    scenes.static_performer()
    result, message = ui._run_and_report("scalar", False)
    if result is not False or "nothing to process" not in message:
        raise Failure("returned %r and said %r" % (result, message))
    return message


def _failed():
    scene = scenes.move_hold_move_hold()

    def broken(writes):
        raise RuntimeError("the keys could not be written")

    real = maya_keys.write_keys
    maya_keys.write_keys = broken
    try:
        result, message = ui._run_and_report("vector", True)
    finally:
        maya_keys.write_keys = real
    if result is not None or "FAILED" not in message or "nothing to process" in message:
        raise Failure("returned %r and said %r" % (result, message))
    for attr in scene.keys:
        common.expect_sparse(scene, scene.control, attr)
    return message
