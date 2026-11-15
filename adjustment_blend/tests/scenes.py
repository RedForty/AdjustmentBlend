"""Scenes for trying and testing Adjustment Blend.

Every function adds its nodes to the current scene and returns a Scene. The tests call them in a
new scene. To look at one yourself, open a new scene in Maya and run, in the Script Editor:

    from adjustment_blend.tests import scenes
    scenes.move_hold_move_hold()

The control and the adjustment layer are left selected, so the Adjustment Blend button works on
the scene as it is.

The performance every scene is built from: the control moves, holds still, moves again, then holds
still to the end. The adjustment layer has a key that changes nothing at the first frame and a key
that offsets the pose at the last frame.
"""
import random
from dataclasses import dataclass, field

import maya.cmds as cmds

START, END = 0, 40
BEATS = (0, 10, 20, 30, 40)     # move from beat 0 to 1, hold to 2, move to 3, hold to 4
HOLDS = ((10, 20), (30, 40))

TRANSFORM_ATTRS = ["translateX", "translateY", "translateZ",
                   "rotateX", "rotateY", "rotateZ",
                   "scaleX", "scaleY", "scaleZ"]


@dataclass
class Scene:
    controls: list
    layer: str                                    # the adjustment layer
    keys: dict = field(default_factory=dict)      # attr -> {time: value}, as keyed on the adjustment layer

    @property
    def control(self):
        return self.controls[0]


# =============================================================================
# Building blocks
# =============================================================================

def performer(name="mover", motion=None, beats=BEATS):
    """A cube that moves, holds, moves, holds.

    motion maps an attribute to (value after the first move, value after the second move).
    The default moves along X and turns around Y.
    """
    if motion is None:
        motion = {"translateX": (10.0, 25.0), "rotateY": (30.0, 75.0)}
    control = cmds.polyCube(name=name, constructionHistory=False)[0]
    for attr, (first, second) in motion.items():
        rest = cmds.getAttr("%s.%s" % (control, attr))
        for time, value in zip(beats, (rest, first, first, second, second)):
            cmds.setKeyframe(control, attribute=attr, time=time, value=value,
                             inTangentType="flat", outTangentType="flat")
    return control


def layer(name, plugs, override=False):
    """A new animation layer on top of the stack, with the plugs as members."""
    name = cmds.animLayer(name, override=override)
    cmds.animLayer(name, edit=True, attribute=plugs)
    return name


def layer_curve(layer_name, plug):
    """The layer's own curve for a plug, or None."""
    curve = cmds.animLayer(layer_name, query=True, findCurveForPlug=plug)
    if not curve:
        return None
    return curve[0] if isinstance(curve, (list, tuple)) else curve


def layer_key(layer_name, control, attr, time, value):
    """Key the layer's own curve: value is what the layer contributes, not the final pose."""
    cmds.setKeyframe(control, attribute=attr, animLayer=layer_name, time=time,
                     inTangentType="flat", outTangentType="flat")
    curve = layer_curve(layer_name, "%s.%s" % (control, attr))
    cmds.keyframe(curve, edit=True, absolute=True, time=(time,), valueChange=value)


def adjustment(controls, offsets, name="Adjust"):
    """The adjustment layer, with sparse keys.

    offsets maps an attribute to the offset at the last frame (a key that changes nothing is set
    at the first frame), or to {time: value} for other key layouts.
    """
    keys = {}
    for attr, offset in offsets.items():
        keys[attr] = dict(offset) if isinstance(offset, dict) else {START: 0.0, END: offset}
    name = layer(name, ["%s.%s" % (c, a) for c in controls for a in keys])
    for control in controls:
        for attr, by_time in keys.items():
            for time, value in sorted(by_time.items()):
                layer_key(name, control, attr, time, value)
    scene = Scene(controls=list(controls), layer=name, keys=keys)
    select(scene)
    return scene


def select(scene):
    """Select the controls and the adjustment layer, as the artist would before blending."""
    cmds.select(scene.controls)
    for each in cmds.ls(type="animLayer"):
        chosen = each == scene.layer
        cmds.animLayer(each, edit=True, selected=chosen, preferred=chosen)


# =============================================================================
# Scenes
# =============================================================================

def move_hold_move_hold():
    """The standard case: the final position and orientation are offset on every axis.

    The performance moves along X and turns around Y only, so the offsets on the other four axes
    have no motion of their own to follow: Smart mode and the Vector signal exist for those.
    """
    control = performer()
    return adjustment([control], {"translateX": 6.0, "translateY": 4.0, "translateZ": -3.0,
                                  "rotateX": 15.0, "rotateY": 20.0, "rotateZ": -10.0})


def translation_only():
    """The performance only translates and the adjustment only rotates.

    The rotation has to borrow the motion of another channel. The control rests at a rotation
    that is not zero and is keyed there, as a posed control is.
    """
    control = performer(motion={"translateX": (10.0, 25.0)})
    cmds.setAttr(control + ".rotate", 20.0, 35.0, -15.0)
    cmds.setKeyframe(control, attribute="rotate", time=START)
    return adjustment([control], {"rotateX": 15.0, "rotateY": 20.0, "rotateZ": -10.0})


def two_performers():
    """Two controls on one adjustment layer that move and hold at different times."""
    first = performer("mover")
    second = performer("late_mover", beats=(0, 5, 25, 35, 40))
    return adjustment([first, second], {"translateX": 6.0, "rotateY": 20.0})


def baked_performer(noise=1e-9, seed=3):
    """The standard case after a bake: a key on every frame of every axis, with rounding noise.

    The four axes that do not move are keyed at their rest value plus or minus the noise, as baked
    and imported animation is. The offsets are the same as in move_hold_move_hold().
    """
    control = performer()
    rng = random.Random(seed)
    attrs = ["translateX", "translateY", "translateZ", "rotateX", "rotateY", "rotateZ"]
    baked = {attr: [cmds.getAttr("%s.%s" % (control, attr), time=frame) for frame in range(START, END + 1)]
             for attr in attrs}
    for attr in attrs:
        for frame, value in zip(range(START, END + 1), baked[attr]):
            cmds.setKeyframe(control, attribute=attr, time=frame, value=value + rng.uniform(-noise, noise))
    return adjustment([control], {"translateX": 6.0, "translateY": 4.0, "translateZ": -3.0,
                                  "rotateX": 15.0, "rotateY": 20.0, "rotateZ": -10.0})


def static_performer():
    """A control with no animation below the adjustment layer."""
    control = cmds.polyCube(name="statue", constructionHistory=False)[0]
    return adjustment([control], {"translateX": 6.0, "rotateY": 20.0})


def scaling_performer():
    """The performance scales, and the adjustment (a multiply layer, Maya's default) scales it more."""
    control = performer(motion={"scaleY": (1.5, 2.0)})
    return adjustment([control], {"scaleY": {START: 1.0, END: 1.5}})


def with_other_layer(position, override=False, mute=False, weight=1.0):
    """The standard performance plus one more layer that moves X during the first hold.

    position is "below" or "above" the adjustment layer.
    """
    control = performer()
    plug = control + ".translateX"

    def other():
        name = layer("Polish", [plug], override=override)
        for time, value in ((START, 0.0), (12, 0.0), (18, 3.0), (END, 3.0)):
            layer_key(name, control, "translateX", time, value)
        cmds.animLayer(name, edit=True, mute=mute, weight=weight)

    if position == "below":
        other()
    scene = adjustment([control], {"translateX": 6.0})
    if position == "above":
        other()
        select(scene)
    return scene


def layered_stack(controls=2, last_frame=12, below=("Low1", "Low2", "Low3"), above=("Top",),
                  group=False, seed=7):
    """Controls with different animation on the base, on the layers below the adjustment layer
    and on the layers above it. With group, the lowest layer sits inside a group layer.

    Returns (scene, the layers above the adjustment layer).
    """
    rng = random.Random(seed)
    names = [cmds.createNode("transform", name="ctrl%d" % i) for i in range(controls)]
    plugs = ["%s.%s" % (c, a) for c in names for a in TRANSFORM_ATTRS]
    times = sorted(set(range(START, last_frame + 1, 4)) | {last_frame})

    def key(layer_name):
        for time in times:
            for attr in TRANSFORM_ATTRS:
                value = (1.0 + rng.uniform(-0.3, 0.3) if attr.startswith("scale")
                         else rng.uniform(-40.0, 40.0) if attr.startswith("rotate")
                         else rng.uniform(-5.0, 5.0))
                extra = {"animLayer": layer_name} if layer_name else {}
                cmds.setKeyframe(names, attribute=attr, time=time, value=value, **extra)

    key(None)
    if group:
        cmds.animLayer("Group")
    for name in below:
        layer(name, plugs)
        key(name)
    if group:
        cmds.animLayer(below[0], edit=True, parent="Group")
    target = layer("Adjust", plugs)
    key(target)
    for name in above:
        layer(name, plugs)
        key(name)
    scene = Scene(controls=names, layer=target)
    select(scene)
    return scene, list(above)
