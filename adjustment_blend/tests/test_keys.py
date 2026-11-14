"""Group "keys": the bulk key writer and its undo.

maya_keys.write_keys must leave a curve as cmds.setKeyframe on every frame would, and the
adjustmentBlendKeys command must make the whole write one undo step that restores the artist's
keys exactly, tangents included.
"""
import maya.cmds as cmds

import adjustment_blend
from adjustment_blend import maya_keys

from . import common, scenes
from .common import Failure

FRAMES = common.FRAMES
VALUES = [0.1 * f + 0.003 * f * f for f in FRAMES]
TOLERANCE = 1e-6


def run(report):
    def check(name, function):
        common.check(report, "keys, " + name, function)

    cmds.undoInfo(state=True)
    for name, shape in SHAPES:
        check("same keys as cmds.setKeyframe, %s" % name, _same_as_setkeyframe(shape))
        check("undo and redo, %s" % name, _undo_redo(shape))
    check("an artist's key keeps its tangent lock and its breakdown mark", _keeps_lock_and_breakdown)
    check("a blend is one undo step", _blend_is_one_undo)
    check("two blends take two undos", _two_blends)
    check("values in meters and radians", _other_units)
    check("a write that fails changes nothing", _failure_changes_nothing)
    check("the command refuses to run by hand", _by_hand)


# =============================================================================
# The artist's keys, in the shapes the writer has to respect
# =============================================================================

def _default(curve):
    pass


def _linear(curve):
    cmds.keyTangent(curve, inTangentType="linear", outTangentType="linear")


def _flat_and_stepped(curve):
    cmds.keyTangent(curve, time=(scenes.START,), inTangentType="flat", outTangentType="step")


def _fixed(curve):
    cmds.keyTangent(curve, inTangentType="fixed", outTangentType="fixed")
    cmds.keyTangent(curve, time=(scenes.START,), edit=True, absolute=True, inAngle=33.0, outAngle=33.0)
    cmds.keyTangent(curve, time=(scenes.END,), edit=True, absolute=True, inAngle=-20.0, outAngle=-20.0)


def _weighted(curve):
    cmds.keyTangent(curve, edit=True, weightedTangents=True)
    cmds.keyTangent(curve, time=(scenes.START,), edit=True, weightLock=False)
    cmds.keyTangent(curve, time=(scenes.START,), inTangentType="fixed", outTangentType="fixed")
    cmds.keyTangent(curve, time=(scenes.START,), edit=True, absolute=True, outAngle=50.0, outWeight=7.0)


def _more_keys(curve):
    """Keys before and after the frames, and one between two frames."""
    cmds.setKeyframe(curve, time=(-10,), value=1.0)
    cmds.setKeyframe(curve, time=(50,), value=2.0)
    cmds.keyTangent(curve, time=(-10,), inTangentType="fixed", outTangentType="fixed")
    cmds.keyTangent(curve, time=(-10,), edit=True, absolute=True, outAngle=25.0)
    cmds.setKeyframe(curve, time=(12.5,), value=3.0)


def _every_frame(curve):
    for frame in FRAMES:
        cmds.setKeyframe(curve, time=(frame,), value=0.01 * frame)
    cmds.keyTangent(curve, time=(7,), inTangentType="linear", outTangentType="flat")


SHAPES = [
    ("default tangents", _default),
    ("linear tangents", _linear),
    ("flat and stepped tangents", _flat_and_stepped),
    ("fixed tangents", _fixed),
    ("weighted tangents", _weighted),
    ("keys outside and between the frames", _more_keys),
    ("a key on every frame already", _every_frame),
]


# =============================================================================
# Reading curves
# =============================================================================

def _curves(shape):
    """The standard scene's translateX and rotateY layer curves, with the artist's keys shaped."""
    scene = scenes.move_hold_move_hold()
    curves = [scenes.layer_curve(scene.layer, "%s.%s" % (scene.control, attr))
              for attr in ("translateX", "rotateY")]
    for curve in curves:
        shape(curve)
    return scene, curves


def _snapshot(curve):
    """Everything about a curve's keys."""
    def tangent(**flag):
        return cmds.keyTangent(curve, query=True, **flag)
    return {
        "times": cmds.keyframe(curve, query=True, timeChange=True),
        "values": cmds.keyframe(curve, query=True, valueChange=True),
        "breakdowns": cmds.keyframe(curve, query=True, breakdown=True) or [],
        "in tangent types": tangent(inTangentType=True),
        "out tangent types": tangent(outTangentType=True),
        "in angles": tangent(inAngle=True),
        "out angles": tangent(outAngle=True),
        "in weights": tangent(inWeight=True),
        "out weights": tangent(outWeight=True),
        "tangent locks": tangent(lock=True),
        "weight locks": tangent(weightLock=True),
    }


def _expect_same(what, want, got, skip=()):
    for name in want:
        if name in skip:
            continue
        a, b = want[name], got[name]
        if len(a) != len(b):
            raise Failure("%s: %d %s, expected %d" % (what, len(b), name, len(a)))
        for i, (x, y) in enumerate(zip(a, b)):
            differs = abs(x - y) > TOLERANCE if isinstance(x, float) else x != y
            if differs:
                raise Failure("%s: %s differ at key %d, %s and not %s" % (what, name, i, y, x))


def _write(curves):
    maya_keys.write_keys([(curve, VALUES) for curve in curves], FRAMES)


# =============================================================================
# Checks
# =============================================================================

def _same_as_setkeyframe(shape):
    def test():
        scene, curves = _curves(shape)
        for curve in curves:
            for frame, value in zip(FRAMES, VALUES):
                cmds.setKeyframe(curve, animLayer=scene.layer, time=(frame,), value=value)
        want = [_snapshot(curve) for curve in curves]

        common.new_scene()
        scene, curves = _curves(shape)
        _write(curves)
        for curve, each in zip(curves, want):
            # cmds.setKeyframe locks the tangents of a key it lands on; the writer leaves them.
            _expect_same(curve, each, _snapshot(curve), skip=("tangent locks",))
        return "%d keys per curve, same values and tangents" % len(want[0]["times"])
    return test


def _undo_redo(shape):
    def test():
        scene, curves = _curves(shape)
        before = [_snapshot(curve) for curve in curves]
        _write(curves)
        after = [_snapshot(curve) for curve in curves]
        if after[0]["values"] == before[0]["values"]:
            raise Failure("the write changed nothing")
        cmds.undo()
        for curve, each in zip(curves, before):
            _expect_same("after undo, " + curve, each, _snapshot(curve))
        cmds.redo()
        for curve, each in zip(curves, after):
            _expect_same("after redo, " + curve, each, _snapshot(curve))
        cmds.undo()
        for curve, each in zip(curves, before):
            _expect_same("after the second undo, " + curve, each, _snapshot(curve))
        return "undo restores %d keys exactly, redo writes %d again" % (
            len(before[0]["times"]), len(after[0]["times"]))
    return test


def _keeps_lock_and_breakdown():
    scene, curves = _curves(_default)
    curve = curves[0]
    cmds.keyTangent(curve, time=(scenes.END,), edit=True, lock=False)
    cmds.setKeyframe(curve, time=(20,), value=3.5, breakdown=True)
    _write(curves)
    if cmds.keyTangent(curve, time=(scenes.END,), query=True, lock=True) != [False]:
        raise Failure("the tangents of the key at frame %d got locked" % scenes.END)
    if cmds.keyframe(curve, query=True, breakdown=True) != [20.0]:
        raise Failure("the breakdown at frame 20 is now an ordinary key")
    got = cmds.keyframe(curve, query=True, time=(20,), valueChange=True)[0]
    if abs(got - VALUES[20]) > TOLERANCE:
        raise Failure("the breakdown at frame 20 holds %.4f, not %.4f" % (got, VALUES[20]))
    return "both kept, with the new values"


def _layer_snapshots(scene):
    return {curve: _snapshot(curve) for curve in cmds.animLayer(scene.layer, query=True, animCurves=True)}


def _blend_is_one_undo():
    scene = scenes.move_hold_move_hold()
    before = _layer_snapshots(scene)
    if not common.blend(scene, smart=True):
        raise Failure("run() returned False")
    after = _layer_snapshots(scene)
    cmds.undo()
    for curve, each in before.items():
        _expect_same("after undo, " + curve, each, _snapshot(curve))
    cmds.redo()
    for curve, each in after.items():
        _expect_same("after redo, " + curve, each, _snapshot(curve))
    for attr in scene.keys:
        common.expect_holds(scene.control, attr)
    return "one undo restores all %d curves, one redo blends them again" % len(before)


def _two_blends():
    scene = scenes.move_hold_move_hold()
    sparse = _layer_snapshots(scene)
    common.blend(scene, smart=True)
    blended = _layer_snapshots(scene)
    common.blend(scene, signal="vector", smart=True)
    cmds.undo()
    for curve, each in blended.items():
        _expect_same("after one undo, " + curve, each, _snapshot(curve))
    cmds.undo()
    for curve, each in sparse.items():
        _expect_same("after two undos, " + curve, each, _snapshot(curve))
    return "each undo steps back one blend"


def _other_units():
    cmds.currentUnit(linear="m", angle="rad")
    try:
        scene, curves = _curves(_default)
        _write(curves)
        for curve in curves:
            got = cmds.keyframe(curve, query=True, valueChange=True)
            worst = max(abs(g - v) for g, v in zip(got, VALUES))
            if len(got) != len(VALUES) or worst > TOLERANCE:
                raise Failure("%s is off by %.3g in meters and radians" % (curve, worst))
    finally:
        cmds.currentUnit(linear="cm", angle="deg")
    return "the values are read as UI units"


def _failure_changes_nothing():
    scene, curves = _curves(_default)
    before = [_snapshot(curve) for curve in curves]
    cmds.select(clear=True)  # the newest entry on the undo queue
    queue = cmds.undoInfo(query=True, undoName=True)
    try:
        maya_keys.write_keys([(curves[0], VALUES), ("no_such_curve", VALUES), (curves[1], VALUES)], FRAMES)
    except Exception as e:  # noqa: BLE001 - any error will do, as long as there is one
        error = type(e).__name__
    else:
        raise Failure("no error for a curve that does not exist")
    for curve, each in zip(curves, before):
        _expect_same("after the failed write, " + curve, each, _snapshot(curve))
    if cmds.undoInfo(query=True, undoName=True) != queue:
        raise Failure("the failed write is on the undo queue")
    try:
        maya_keys.write_keys([(curves[0], VALUES[:5])], FRAMES)
    except ValueError:
        pass
    else:
        raise Failure("no error for 5 values on %d frames" % len(FRAMES))
    return "%s raised, the curve written before it is restored, nothing to undo" % error


def _by_hand():
    scene, curves = _curves(_default)
    _write(curves)  # loads the plug-in
    before = [_snapshot(curve) for curve in curves]
    try:
        getattr(cmds, maya_keys.COMMAND)()
    except Exception:  # noqa: BLE001
        pass
    else:
        raise Failure("%s ran with nothing to record" % maya_keys.COMMAND)
    cmds.undo()  # must undo the write, not a failed command
    if _snapshot(curves[0])["times"] == before[0]["times"]:
        raise Failure("undo after the refused command did not undo the write before it")
    return "it raises and leaves the undo queue alone"
