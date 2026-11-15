"""Group "blend": what Adjustment Blend does to the adjustment layer, case by case.

The standard case is scenes.move_hold_move_hold(): a control moves, holds, moves and holds, and the
artist offsets its final position and orientation on an additive layer with one key at each end.
Left alone, the two sparse keys make the control drift through its holds. After the blend the layer
is keyed on every frame, the offset arrives in step with the original motion, and the holds hold.
"""
import maya.cmds as cmds

import adjustment_blend

from . import common, scenes
from .common import Failure

MOVING = ["translateX", "rotateY"]                                # axes the standard performance moves
STILL = ["translateY", "translateZ", "rotateX", "rotateZ"]        # axes it does not
VECTOR_TOLERANCE = 1e-4                                           # angular speed goes through acos


def run(report):
    def check(name, function, **more):
        common.check(report, "blend, " + name, function, **more)

    check("before blending, sparse keys drift through the holds", _drift_before)
    check("standard, scalar normal", _standard_scalar_normal)
    check("standard, scalar smart", _standard_scalar_smart)
    check("standard, vector normal", _standard_vector)
    check("standard, vector smart", lambda: _standard_vector(smart=True))
    check("rotation offset on a translating control, scalar normal", _cross_channel_scalar_normal)
    check("rotation offset on a translating control, scalar smart", lambda: _cross_channel("scalar", True))
    check("rotation offset on a translating control, vector normal", _cross_channel_vector_normal)
    check("rotation offset on a translating control, vector smart", lambda: _cross_channel("vector", True))
    check("scale offset on a scaling control", _scale)
    check("two controls that hold at different times", _two_controls)
    check("another layer below adds motion", lambda: _other_layer("below"))
    check("a muted layer below is ignored", lambda: _other_layer("below", mute=True))
    check("a half-weight layer below counts half", lambda: _other_layer("below", weight=0.5))
    check("an override layer below replaces the base", lambda: _other_layer("below", override=True))
    check("a layer above is ignored", lambda: _other_layer("above"))
    check("a control with no motion is left alone", _static)
    check("one key on the layer is not enough", _one_key)
    check("a control that is not on the layer is refused", _not_a_member)
    check("apply=False writes nothing", _dry_run)
    for signal in ("scalar", "vector"):
        check("a third key in the middle of a move, %s" % signal, lambda: _third_key_mid_move(signal))
        check("two keys span a hold, %s" % signal, lambda: _keys_span_a_hold(signal))


# =============================================================================
# The standard case
# =============================================================================

def _drift_before():
    scene = scenes.move_hold_move_hold()
    control = scene.control
    for attr in MOVING:
        below = common.below_values(scene, control, attr)
        for hold in scenes.HOLDS:
            if common.drift(below, common.FRAMES, hold) > 1e-9:
                raise Failure("the scene is wrong: %s moves during a hold with the layer muted" % attr)
    moved = common.drift(common.final_values(control, "translateX"), common.FRAMES, scenes.HOLDS[0])
    if moved < 0.1:
        raise Failure("translateX only drifts %.4f in the first hold, so the scene shows no problem" % moved)
    return "translateX drifts %.2f during the first hold" % moved


def _standard_scalar_normal():
    scene = scenes.move_hold_move_hold()
    if not common.blend(scene):
        raise Failure("run() returned False")
    for attr in MOVING:
        common.expect_follows(scene, scene.control, attr)
        common.expect_holds(scene.control, attr)
    for attr in STILL:
        common.expect_sparse(scene, scene.control, attr)
    return "the moving axes hold; the still axes keep their two keys"


def _standard_scalar_smart():
    scene = scenes.move_hold_move_hold()
    if not common.blend(scene, smart=True):
        raise Failure("run() returned False")
    borrowed = {"translateY": "translateX", "translateZ": "translateX", "rotateX": "rotateY", "rotateZ": "rotateY"}
    for attr in MOVING + STILL:
        common.expect_follows(scene, scene.control, attr, source=borrowed.get(attr))
        common.expect_holds(scene.control, attr)
    return "all six axes hold; the still axes follow the moving axis of their channel"


def _standard_vector(smart=False):
    scene = scenes.move_hold_move_hold()
    if not common.blend(scene, signal="vector", smart=smart):
        raise Failure("run() returned False")
    for attr in MOVING + STILL:
        source = "translateX" if attr.startswith("translate") else "rotateY"
        common.expect_follows(scene, scene.control, attr, source=source, tolerance=VECTOR_TOLERANCE)
        common.expect_holds(scene.control, attr, tolerance=VECTOR_TOLERANCE)
    return "all six axes hold; each channel shares one speed"


# =============================================================================
# Borrowing motion from another channel
# =============================================================================

ROTATION = ["rotateX", "rotateY", "rotateZ"]


def _cross_channel_scalar_normal():
    scene = scenes.translation_only()
    if common.blend(scene):
        raise Failure("run() returned True with no rotation below to follow")
    for attr in ROTATION:
        common.expect_sparse(scene, scene.control, attr)
    return "nothing to follow, nothing changed"


def _cross_channel_vector_normal():
    scene = scenes.translation_only()
    common.blend(scene, signal="vector")
    for attr in ROTATION:
        common.expect_sparse(scene, scene.control, attr)
    return "nothing to follow, nothing changed"


def _cross_channel(signal, smart):
    scene = scenes.translation_only()
    if not common.blend(scene, signal=signal, smart=smart):
        raise Failure("run() returned False")
    for attr in ROTATION:
        common.expect_follows(scene, scene.control, attr, source="translateX", tolerance=VECTOR_TOLERANCE)
        common.expect_holds(scene.control, attr, tolerance=VECTOR_TOLERANCE)
    return "the rotation follows the translation and holds"


# =============================================================================
# Other shapes of the standard case
# =============================================================================

def _scale():
    scene = scenes.scaling_performer()
    if not common.blend(scene):
        raise Failure("run() returned False")
    common.expect_follows(scene, scene.control, "scaleY")
    common.expect_holds(scene.control, "scaleY")
    return "scaleY follows the scaling below and holds"


def _two_controls():
    scene = scenes.two_performers()
    if not common.blend(scene):
        raise Failure("run() returned False")
    for control, holds in zip(scene.controls, (scenes.HOLDS, ((5, 25), (35, 40)))):
        for attr in MOVING:
            common.expect_follows(scene, control, attr)
            common.expect_holds(control, attr, holds=holds)
    return "each control follows its own motion"


def _other_layer(position, **settings):
    scene = scenes.with_other_layer(position, **settings)
    above = ["Polish"] if position == "above" else []
    if not common.blend(scene):
        raise Failure("run() returned False")
    common.expect_follows(scene, scene.control, "translateX", above=above)
    return "translateX follows what Maya shows below the adjustment layer"


# =============================================================================
# Nothing to do
# =============================================================================

def _static():
    for options in (dict(), dict(smart=True), dict(signal="vector"), dict(signal="vector", smart=True)):
        common.new_scene()
        scene = scenes.static_performer()
        if common.blend(scene, **options):
            raise Failure("run(%s) returned True for a control that never moves" % options)
        for attr in scene.keys:
            common.expect_sparse(scene, scene.control, attr)
    return "left with its two keys in every mode"


def _one_key():
    control = scenes.performer()
    scene = scenes.adjustment([control], {"translateX": {scenes.END: 6.0}})
    if common.blend(scene):
        raise Failure("run() returned True with one key on the layer")
    common.expect_sparse(scene, control, "translateX")
    return "refused, nothing changed"


def _not_a_member():
    scene = scenes.move_hold_move_hold()
    bystander = scenes.performer("bystander")
    root = cmds.animLayer(query=True, root=True)
    if adjustment_blend.run(scene.layer, [root], [bystander]):
        raise Failure("run() returned True for a control that is not on the layer")
    for attr in scene.keys:
        common.expect_sparse(scene, scene.control, attr)
    return "refused, nothing changed"


def _dry_run():
    scene = scenes.move_hold_move_hold()
    if not common.blend(scene, smart=True, apply=False):
        raise Failure("run(apply=False) returned False")
    for attr in scene.keys:
        common.expect_sparse(scene, scene.control, attr)
    return "returned True and left the two keys"


# =============================================================================
# More than two keys
# =============================================================================

def _third_key_mid_move(signal):
    control = scenes.performer()
    scene = scenes.adjustment([control], {"translateX": {0: 0.0, 25: 4.0, 40: 6.0}})
    if not common.blend(scene, signal=signal):
        raise Failure("run() returned False")
    common.expect_keys_kept(scene, control, "translateX")
    common.expect_follows(scene, control, "translateX", tolerance=VECTOR_TOLERANCE)
    return "both spans follow the motion below"


def _keys_span_a_hold(signal):
    """The control holds still from frame 10 to 20, and the artist keys a change across that hold.

    There is no motion to follow there, so the change is spread evenly. The span after it follows
    the motion as usual.
    """
    control = scenes.performer()
    keys = {10: 0.0, 20: 5.0, 40: 6.0}
    scene = scenes.adjustment([control], {"translateX": keys})
    if not common.blend(scene, signal=signal):
        raise Failure("run() returned False")
    common.expect_keys_kept(scene, control, "translateX")
    frames = list(range(10, 41))
    got = common.layer_values(scene, control, "translateX", frames)
    below = common.below_values(scene, control, "translateX")
    want = [0.5 * (f - 10) for f in range(10, 20)] + common.expected({20: 5.0, 40: 6.0}, below)
    worst = max(range(len(frames)), key=lambda i: abs(got[i] - want[i]))
    if abs(got[worst] - want[worst]) > VECTOR_TOLERANCE:
        raise Failure("translateX is %.4f at frame %d and must be %.4f" % (got[worst], frames[worst], want[worst]))
    return "the keys keep their values; even steps through the hold, then it follows the motion"
