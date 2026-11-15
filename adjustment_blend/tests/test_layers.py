"""Group "layers": LayerStack reads the same values below the adjustment layer as Maya does.

Maya's own answer comes from muting the adjustment layer and every layer above it and reading the
attribute. LayerStack must match it for every layer setting, on translate, rotate and scale.
"""
import maya.cmds as cmds

from adjustment_blend.maya_layers import LayerStack

from . import common, scenes
from .common import Failure

LAST = 12
FRAMES = list(range(scenes.START, LAST + 1))
TOLERANCE = 1e-9


def _edit(layer, **flags):
    return lambda: cmds.animLayer(layer, edit=True, **flags)


def _both(*steps):
    return lambda: [step() for step in steps]


def _set(plug, value):
    return lambda: cmds.setAttr(plug, value)


def _animate(plug, first, last):
    def keys():
        cmds.setKeyframe(plug, time=scenes.START, value=first)
        cmds.setKeyframe(plug, time=LAST, value=last)
    return keys


def _negative_scale():
    """A negative scale on a layer with a fractional weight: curve math cannot follow, Maya must."""
    for curve in cmds.animLayer("Low1", query=True, animCurves=True):
        if "scaleX" in curve:
            cmds.keyframe(curve, edit=True, absolute=True, valueChange=-1.5)
    cmds.animLayer("Low1", edit=True, weight=0.5)


SETTINGS = [  # name, what to change in scenes.layered_stack(), options for layered_stack()
    ("additive layers", None, {}),
    ("an override layer", _edit("Low2", override=True), {}),
    ("an override layer at weight 0.3 under a layer at 0.6",
     _both(_edit("Low2", override=True, weight=0.3), _edit("Low3", weight=0.6)), {}),
    ("an override layer with passthrough off", _edit("Low2", override=True, passthrough=False, weight=0.3), {}),
    ("a muted layer", _edit("Low1", mute=True), {}),
    ("a soloed layer", _edit("Low1", solo=True), {}),
    ("an animated layer weight", _animate("Low3.weight", 0.0, 1.0), {}),
    ("an animated layer mute", _animate("Low1.mute", 0, 1), {}),
    ("scale added, at weight 0.5", _both(_set("Low1.scaleAccumulationMode", 0), _edit("Low1", weight=0.5)), {}),
    ("scale multiplied, at weight 0.5", _edit("Low1", weight=0.5), {}),
    ("rotation accumulated as quaternions", _set("Low2.rotationAccumulationMode", 1), {}),
    ("a negative scale at weight 0.5", _negative_scale, {}),
    ("a group layer at weight 0.5", _edit("Group", weight=0.5), dict(group=True)),
    ("a muted group layer", _edit("Group", mute=True), dict(group=True)),
    ("an animated group layer weight", _animate("Group.weight", 0.2, 0.9), dict(group=True)),
    ("the adjustment layer on top", None, dict(above=())),
    ("the adjustment layer directly on the base", None, dict(below=())),
]


def run(report):
    def check(name, function):
        common.check(report, "layers, " + name, function)

    for name, change, options in SETTINGS:
        check(name, _matches_maya(change, options))
    check("an attribute that is not on the adjustment layer", _not_on_the_layer)
    check("an attribute with a curve and no layers", _curve_only)
    check("attributes with nothing to read", _nothing_to_read)
    check("curve_values reads a curve as cmds.keyframe does", _curve_values)


def _matches_maya(change, options):
    def test():
        scene, above = scenes.layered_stack(last_frame=LAST, **options)
        if change:
            change()
        stack = LayerStack(scene.layer, FRAMES)
        worst = 0.0
        for control in scene.controls:
            for attr in scenes.TRANSFORM_ATTRS:
                got = stack.base_values(control, attr)
                if got is None:
                    raise Failure("no values for %s.%s" % (control, attr))
                want = common.below_values(scene, control, attr, above, FRAMES)
                difference = max(abs(g - w) for g, w in zip(got, want))
                if difference > TOLERANCE:
                    raise Failure("%s.%s differs from Maya by %.3g" % (control, attr, difference))
                worst = max(worst, difference)
        return "largest difference from Maya %.1e" % worst
    return test


def _not_on_the_layer():
    control = scenes.performer(motion={"translateX": (10.0, 25.0), "translateY": (2.0, 5.0)})
    low = scenes.layer("Low", [control + ".translateX", control + ".translateY"])
    for attr in ("translateX", "translateY"):
        scenes.layer_key(low, control, attr, scenes.START, 0.0)
        scenes.layer_key(low, control, attr, LAST, 3.0)
    scene = scenes.adjustment([control], {"translateX": 6.0})
    got = LayerStack(scene.layer, FRAMES).base_values(control, "translateY")
    want = common.final_values(control, "translateY", FRAMES)
    difference = max(abs(g - w) for g, w in zip(got, want))
    if difference > TOLERANCE:
        raise Failure("translateY differs from its layered value by %.3g" % difference)
    return "reads the whole layered value"


def _curve_only():
    control = scenes.performer()
    stack = LayerStack("Adjust", FRAMES)
    for attr in ("translateX", "ry"):
        got = stack.base_values(control, attr)
        want = common.final_values(control, attr, FRAMES)
        difference = max(abs(g - w) for g, w in zip(got, want))
        if difference > TOLERANCE:
            raise Failure("%s differs from its curve by %.3g" % (attr, difference))
    return "reads the curve; rotation in degrees; short names work"


def _curve_values():
    scene = scenes.move_hold_move_hold()
    stack = LayerStack(scene.layer, common.FRAMES)
    for attr in ("translateX", "rotateY"):
        curve = scenes.layer_curve(scene.layer, "%s.%s" % (scene.control, attr))
        got = stack.curve_values(curve)
        want = common.layer_values(scene, scene.control, attr)
        difference = max(abs(g - w) for g, w in zip(got, want))
        if difference > TOLERANCE:
            raise Failure("%s differs from cmds.keyframe by %.3g" % (attr, difference))
    return "translation in centimeters, rotation in degrees"


def _nothing_to_read():
    control = scenes.performer()
    driver = scenes.performer("driver")
    constrained = cmds.createNode("transform", name="constrained")
    cmds.setKeyframe(constrained, attribute="translateX", time=scenes.START, value=0.0)
    cmds.parentConstraint(driver, constrained)
    stack = LayerStack("Adjust", FRAMES)
    cases = [("a static attribute", control, "translateZ"),
             ("a missing attribute", control, "nope"),
             ("a missing node", "ghost", "translateX"),
             ("a constrained attribute", constrained, "translateX")]
    for name, node, attr in cases:
        got = stack.base_values(node, attr)
        if got is not None:
            raise Failure("%s gave %d values, not None" % (name, len(got)))
    return "None for static, missing and constrained attributes"
