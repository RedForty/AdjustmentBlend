"""Group "blend": what Adjustment Blend does to the adjustment layer, case by case.

The standard case is scenes.move_hold_move_hold(): a control moves, holds, moves and holds, and the
artist offsets its final position and orientation on an additive layer with one key at each end.
Left alone, the two sparse keys make the control drift through its holds. After the blend the layer
is keyed on every frame, the offset arrives in step with the original motion, and the holds hold.
"""
import maya.cmds as cmds

import adjustment_blend
from adjustment_blend import maya_scene

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
    check("baked keys with rounding noise, scalar normal", _baked_scalar_normal)
    check("baked keys with rounding noise, scalar smart", lambda: _baked("scalar"))
    check("baked keys with rounding noise, vector", lambda: _baked("vector"))
    check("scale offset on a scaling control", _scale)
    check("two controls that hold at different times", _two_controls)
    for signal in ("scalar", "vector"):
        check("two controls keyed on different frames, %s" % signal, lambda: _own_keys_two_controls(signal))
        check("two attributes keyed on different frames, %s" % signal, lambda: _own_keys_one_control(signal))
    check("a first key between two frames", _key_between_frames)
    check("two controls with the same name", _same_name)
    check("another layer below adds motion", lambda: _other_layer("below"))
    check("a muted layer below is ignored", lambda: _other_layer("below", mute=True))
    check("a half-weight layer below counts half", lambda: _other_layer("below", weight=0.5))
    check("an override layer below replaces the base", lambda: _other_layer("below", override=True))
    check("a layer above is ignored", lambda: _other_layer("above"))
    check("with two layers selected, the upper one is blended", _two_layers_selected)
    check("with no layer selected, the top one is blended", _no_layer_selected)
    check("the stack order follows moved and nested layers", _stack_order)
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
# Baked animation: a key on every frame, and noise on the axes that do not move
# =============================================================================

def _baked_scalar_normal():
    scene = scenes.baked_performer()
    if not common.blend(scene):
        raise Failure("run() returned False")
    for attr in MOVING:
        common.expect_follows(scene, scene.control, attr)
        common.expect_holds(scene.control, attr)
    for attr in STILL:
        common.expect_sparse(scene, scene.control, attr)
    return "the noise on the still axes is not taken for motion: they keep their two keys"


def _baked(signal):
    scene = scenes.baked_performer()
    if not common.blend(scene, signal=signal, smart=True):
        raise Failure("run() returned False")
    for attr in MOVING + STILL:
        source = "translateX" if attr.startswith("translate") else "rotateY"
        common.expect_follows(scene, scene.control, attr, source=source, tolerance=VECTOR_TOLERANCE)
        common.expect_holds(scene.control, attr, tolerance=VECTOR_TOLERANCE)
    return "the still axes follow the moving axis of their channel, and all six hold"


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


# =============================================================================
# Every curve is blended between its own keys
# =============================================================================

def _blend_all(made, **options):
    controls = [scene.control for scene in made]
    if not adjustment_blend.run(made[0].layer, controls, **options):
        raise Failure("run() returned False")


def _own_keys_two_controls(signal):
    """One control is keyed to frame 40 and the other to frame 30, where the first one is holding."""
    first = scenes.performer("first")
    second = scenes.performer("second")
    made = scenes.adjustment_by_plug({first + ".translateX": {0: 0.0, 40: 6.0},
                                      second + ".translateX": {0: 0.0, 30: 6.0}})
    _blend_all(made, signal=signal)
    for scene in made:
        # expect_follows also checks the keys stop at the curve's own last key.
        common.expect_follows(scene, scene.control, "translateX", tolerance=VECTOR_TOLERANCE)
        common.expect_holds(scene.control, "translateX", tolerance=VECTOR_TOLERANCE)
    return "each follows the motion between its own keys, and both hold"


def _own_keys_one_control(signal):
    control = scenes.performer()
    made = scenes.adjustment_by_plug({control + ".translateX": {0: 0.0, 40: 6.0},
                                      control + ".rotateY": {0: 0.0, 25: 10.0, 40: 20.0}})
    _blend_all(made, signal=signal)
    for attr in ("translateX", "rotateY"):
        common.expect_follows(made[0], control, attr, tolerance=VECTOR_TOLERANCE)
        common.expect_keys_kept(made[0], control, attr)
    common.expect_holds(control, "translateX", tolerance=VECTOR_TOLERANCE)
    return "the third key on rotateY does not cut translateX in two"


def _key_between_frames():
    """The artist's first key is at frame 0.5. It stays, and no key is set before it."""
    control = scenes.performer()
    scene = scenes.adjustment_by_plug({control + ".translateX": {0.5: 0.0, 40: 6.0}})[0]
    _blend_all([scene])
    times = common.key_times(scene, control, "translateX")
    if times != [0.5] + [float(f) for f in range(1, 41)]:
        raise Failure("keys at %s ... %s, not at 0.5 and then on each frame from 1 to 40" % (times[:3], times[-1]))
    samples = [0.5] + list(range(1, 41))
    below = common.below_values(scene, control, "translateX", frames=samples)
    got = common.layer_values(scene, control, "translateX", samples)
    travelled, total = 0.0, sum(abs(b - a) for a, b in zip(below, below[1:]))
    for i in range(len(samples)):
        if i:
            travelled += abs(below[i] - below[i - 1])
        want = 6.0 * travelled / total
        if abs(got[i] - want) > 1e-6:
            raise Failure("translateX is %.4f at frame %s and must be %.4f" % (got[i], samples[i], want))
    return "keyed at 0.5 and on frames 1 to 40, following the motion from the key on"


def _same_name():
    """Two controls called "hand" under different groups, as the left and right side of a rig can be."""
    controls = []
    for side, beats in (("left", scenes.BEATS), ("right", (0, 5, 25, 35, 40))):
        control = scenes.performer("hand", beats=beats)
        group = cmds.group(empty=True, name=side)
        cmds.parent(control, group)
        controls.append("%s|hand" % side)
    made = scenes.adjustment_by_plug({control + ".translateX": {0: 0.0, 40: 6.0} for control in controls})
    _blend_all(made)
    for scene, holds in zip(made, (scenes.HOLDS, ((5, 25), (35, 40)))):
        common.expect_follows(scene, scene.control, "translateX")
        common.expect_holds(scene.control, "translateX", holds=holds)
    return "%s and %s each follow their own motion" % tuple(controls)


def _other_layer(position, **settings):
    scene = scenes.with_other_layer(position, **settings)
    above = ["Polish"] if position == "above" else []
    if not common.blend(scene):
        raise Failure("run() returned False")
    common.expect_follows(scene, scene.control, "translateX", above=above)
    return "translateX follows what Maya shows below the adjustment layer"


# =============================================================================
# Which layer is the adjustment layer, when run() is not told
# =============================================================================

def _select_layers(*names):
    for each in cmds.ls(type="animLayer"):
        cmds.animLayer(each, edit=True, selected=each in names, preferred=False)


def _two_layers_selected():
    """The names are chosen so that sorting by name would pick the lower layer."""
    control = scenes.performer()
    lower = scenes.adjustment_by_plug({control + ".translateX": {0: 0.0, 40: 2.0}}, name="Zeta")[0]
    upper = scenes.adjustment_by_plug({control + ".translateX": {0: 0.0, 40: 6.0}}, name="Alpha")[0]
    _select_layers("Zeta", "Alpha")
    if maya_scene.discover_adjustment_layer() != "Alpha":
        raise Failure("picked %s, and Alpha is the upper of the two" % maya_scene.discover_adjustment_layer())
    if not adjustment_blend.run(objects=[control]):
        raise Failure("run() returned False")
    common.expect_follows(upper, control, "translateX", above=[])
    common.expect_sparse(lower, control, "translateX")
    return "Alpha is blended and Zeta, below it, keeps its two keys"


def _no_layer_selected():
    scene = scenes.with_other_layer("below")
    _select_layers()
    cmds.select(scene.controls)
    if not adjustment_blend.run():
        raise Failure("run() returned False")
    common.expect_follows(scene, scene.control, "translateX")
    return "%s, the top of the stack, is blended" % scene.layer


def _stack_order():
    control = scenes.performer()
    for name in ("Zeta", "Alpha", "Mid", "Top"):
        scenes.layer(name, [control + ".translateX"])
    steps = [
        ("as created", None, ["Zeta", "Alpha", "Mid", "Top"]),
        ("Top moved to the bottom", lambda: cmds.animLayer("Top", edit=True, moveLayerBefore="Zeta"),
         ["Top", "Zeta", "Alpha", "Mid"]),
        ("Alpha put inside a group layer on top", lambda: (cmds.animLayer("Group"),
                                                         cmds.animLayer("Alpha", edit=True, parent="Group")),
         ["Top", "Zeta", "Mid", "Group", "Alpha"]),
    ]
    for name, change, want in steps:
        if change:
            change()
        got = maya_scene.stack_order()
        if got != want:
            raise Failure("%s: the stack reads %s, not %s" % (name, got, want))
    _select_layers("Zeta", "Top", "Mid")
    if maya_scene.discover_adjustment_layer() != "Mid":
        raise Failure("with Zeta, Top and Mid selected it picked %s, not Mid" % maya_scene.discover_adjustment_layer())
    return "bottom to top after a move and a nesting; the upper of three selected layers is Mid"


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
    if adjustment_blend.run(scene.layer, [bystander]):
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
