"""What the test groups share: the check wrapper, reading the scene, and the expectations."""
import os
import traceback

import maya.cmds as cmds

import adjustment_blend

from . import scenes

FRAMES = list(range(scenes.START, scenes.END + 1))


class Failure(Exception):
    """A check that does not hold. The message becomes the check's detail."""


def new_scene():
    cmds.file(new=True, force=True)
    cmds.currentUnit(linear="cm", angle="deg")


def check(report, name, function, known_bug=None):
    """Run one check in a new scene. It passes when function returns; what it returns is the detail.

    known_bug names a bug that is on record and not fixed: the check is then expected to fail.
    """
    failed = None
    detail = ""
    try:
        new_scene()
        detail = function() or ""
    except Failure as e:
        failed = str(e)
    except Exception as e:
        where = traceback.extract_tb(e.__traceback__)[-1]
        failed = "%s: %s (%s line %d)" % (type(e).__name__, " ".join(str(e).split()),
                                         os.path.basename(where.filename), where.lineno)
    if known_bug:
        if failed is None:
            report(False, name, "passes now, so remove its known_bug marker (%s)" % known_bug)
        else:
            report(None, name, "%s. It shows as: %s" % (known_bug, failed))
    elif failed is None:
        report(True, name, detail)
    else:
        report(False, name, failed)


# =============================================================================
# Running the tool and reading the scene
# =============================================================================

def blend(scene, **options):
    """Run Adjustment Blend on a scene the way a script would, with no UI."""
    root = cmds.animLayer(query=True, root=True)
    return adjustment_blend.run(scene.layer, [root], scene.controls, **options)


def key_times(scene, control, attr):
    curve = scenes.layer_curve(scene.layer, "%s.%s" % (control, attr))
    return cmds.keyframe(curve, query=True, timeChange=True) or []


def layer_values(scene, control, attr, frames=FRAMES):
    """What the adjustment layer's own curve holds at each frame."""
    curve = scenes.layer_curve(scene.layer, "%s.%s" % (control, attr))
    return [cmds.keyframe(curve, query=True, eval=True, valueChange=True, time=(f,))[0] for f in frames]


def final_values(control, attr, frames=FRAMES):
    """The attribute's value at each frame, with every layer applied."""
    return [cmds.getAttr("%s.%s" % (control, attr), time=f) for f in frames]


def below_values(scene, control, attr, above=(), frames=FRAMES):
    """Maya's own value of the attribute below the adjustment layer.

    Found by muting the adjustment layer and the layers above it, then reading the attribute.
    """
    layers = [scene.layer] + list(above)
    was = [cmds.animLayer(each, query=True, mute=True) for each in layers]
    for each in layers:
        cmds.animLayer(each, edit=True, mute=True)
    try:
        return final_values(control, attr, frames)
    finally:
        for each, mute in zip(layers, was):
            cmds.animLayer(each, edit=True, mute=mute)


# =============================================================================
# Expectations
# =============================================================================

def expected(keys, below):
    """What the adjustment curve must hold on every frame from the first key to the last.

    Between two neighboring keys, the change in value is handed out in proportion to how far the
    animation below moves on each frame. below[i] is the value below at frame START + i.
    """
    times = sorted(keys)
    values = {times[0]: keys[times[0]]}
    for first, last in zip(times, times[1:]):
        steps = [abs(below[f - scenes.START] - below[f - 1 - scenes.START]) for f in range(first + 1, last + 1)]
        total = sum(steps)
        if total == 0.0:
            raise Failure("the animation below does not move between frames %d and %d" % (first, last))
        travelled = 0.0
        for frame, step in zip(range(first + 1, last + 1), steps):
            travelled += step
            values[frame] = keys[first] + (keys[last] - keys[first]) * travelled / total
    return [values[f] for f in range(times[0], times[-1] + 1)]


def expect_follows(scene, control, attr, source=None, above=(), tolerance=1e-6):
    """The adjustment curve is keyed on every frame and its value follows the motion below.

    source is the attribute whose motion it must follow, when that is not its own.
    """
    keys = scene.keys[attr]
    frames = list(range(min(keys), max(keys) + 1))
    times = key_times(scene, control, attr)
    if times != [float(f) for f in frames]:
        raise Failure("%s has %d keys, not one on each frame from %d to %d" % (attr, len(times), frames[0], frames[-1]))
    want = expected(keys, below_values(scene, control, source or attr, above))
    got = layer_values(scene, control, attr, frames)
    worst = max(range(len(frames)), key=lambda i: abs(got[i] - want[i]))
    if abs(got[worst] - want[worst]) > tolerance:
        raise Failure("%s is %.4f at frame %d, and following %s it must be %.4f" % (
            attr, got[worst], frames[worst], source or "its own motion", want[worst]))


def expect_keys_kept(scene, control, attr, tolerance=1e-6):
    """The artist's keys on the adjustment layer still hold the values the artist set."""
    for time, value in sorted(scene.keys[attr].items()):
        got = layer_values(scene, control, attr, [time])[0]
        if abs(got - value) > tolerance:
            raise Failure("the key on %s at frame %d was %.4f and is now %.4f" % (attr, time, value, got))


def expect_sparse(scene, control, attr):
    """The adjustment curve still has only the artist's keys, with their values."""
    times = key_times(scene, control, attr)
    if times != [float(t) for t in sorted(scene.keys[attr])]:
        raise Failure("%s has keys at %s, not only the artist's keys" % (attr, times))
    expect_keys_kept(scene, control, attr)


def drift(values, frames, hold):
    """How far a value travels during a hold."""
    held = [v for f, v in zip(frames, values) if hold[0] <= f <= hold[1]]
    return max(held) - min(held)


def expect_holds(control, attr, holds=scenes.HOLDS, tolerance=1e-6):
    """With every layer applied, the attribute stays still during each hold of the performance."""
    values = final_values(control, attr)
    for hold in holds:
        moved = drift(values, FRAMES, hold)
        if moved > tolerance:
            raise Failure("%s moves %.4f during the hold from frame %d to %d" % (attr, moved, hold[0], hold[1]))
