"""
A tiny Maya UI for Adjustment Blend.

One big button plus two toggles — Signal (Scalar / Vector) and Mode
(Normal / Smart) — that call :func:`adjustment_blend.run`. Deliberately thin:
all the real logic lives in the package; this just flips the two flags, wraps
the run in a single undo chunk, and reports the result.

The two toggles persist to ``optionVar`` so a user's choice is remembered across
scenes and sessions.

Built with ``maya.cmds`` UI (rather than Qt) so it works across Maya versions
with no PySide2/PySide6 shims.

    import adjustment_blend
    adjustment_blend.show_ui()
"""

from __future__ import annotations

import functools
import logging

from maya import cmds

from .pipeline import run

log = logging.getLogger(__name__)

_WINDOW = "adjustmentBlendWindow"

# Persistent preferences (1 = first option, 2 = second option).
_OPT_SIGNAL = "adjustmentBlendSignal"   # 1 = Scalar, 2 = Vector
_OPT_MODE = "adjustmentBlendMode"       # 1 = Normal, 2 = Smart


def show():
    """Create (or re-show) the Adjustment Blend window."""
    if cmds.window(_WINDOW, exists=True):
        cmds.deleteUI(_WINDOW)

    # Maya persists a window's last size in its prefs and restores it on
    # recreate — which is why editing the layout size can look like it does
    # nothing, and why a once-tall window stays tall with empty space. Clearing
    # the pref lets the window size itself to its current content.
    if cmds.windowPref(_WINDOW, exists=True):
        cmds.windowPref(_WINDOW, remove=True)

    win = cmds.window(_WINDOW, title="Adjustment Blend",
                      sizeable=False, resizeToFitChildren=True)
    cmds.columnLayout(adjustableColumn=True, rowSpacing=4,
                      columnAttach=("both", 8), width=190)

    cmds.separator(height=3, style="none")

    # The button. Command is wired below, once the toggles exist.
    button = cmds.button(
        label="Adjustment Blend",
        height=46,
        annotation="Blend the selected adjustment layer using the options below.",
    )

    cmds.separator(height=3, style="in")

    signal_ctrl = cmds.radioButtonGrp(
        label="Signal",
        labelArray2=["Scalar", "Vector"],
        numberOfRadioButtons=2,
        select=_load_choice(_OPT_SIGNAL),
        columnWidth3=(48, 62, 62),
        annotation="Scalar: per-attribute velocity.  "
                   "Vector: one shared speed per channel group.",
    )
    mode_ctrl = cmds.radioButtonGrp(
        label="Mode",
        labelArray2=["Normal", "Smart"],
        numberOfRadioButtons=2,
        select=_load_choice(_OPT_MODE),
        columnWidth3=(48, 62, 62),
        annotation="Smart: borrow motion when an attribute has none of its own.",
    )

    cmds.separator(height=3, style="none")

    # Persist each toggle the moment it changes, and wire the run button.
    cmds.radioButtonGrp(signal_ctrl, edit=True,
                        changeCommand=functools.partial(_save_choice, _OPT_SIGNAL, signal_ctrl))
    cmds.radioButtonGrp(mode_ctrl, edit=True,
                        changeCommand=functools.partial(_save_choice, _OPT_MODE, mode_ctrl))
    cmds.button(button, edit=True,
                command=functools.partial(_on_run, signal_ctrl, mode_ctrl))

    cmds.showWindow(win)
    return win


def _load_choice(opt_key, default=1):
    """Read a persisted 1/2 selection, defaulting when unset or corrupt."""
    if cmds.optionVar(exists=opt_key):
        value = cmds.optionVar(q=opt_key)
        if value in (1, 2):
            return value
    return default


def _save_choice(opt_key, ctrl, *_args):
    """Persist a toggle's current selection to its optionVar."""
    cmds.optionVar(intValue=(opt_key, cmds.radioButtonGrp(ctrl, q=True, select=True)))


def _on_run(signal_ctrl, mode_ctrl, *_args):
    """Read the toggles and run, as a single undoable operation."""
    signal = "vector" if cmds.radioButtonGrp(signal_ctrl, q=True, select=True) == 2 else "scalar"
    smart = cmds.radioButtonGrp(mode_ctrl, q=True, select=True) == 2

    cmds.undoInfo(openChunk=True, chunkName="Adjustment Blend")
    try:
        result = run(signal=signal, smart=smart)
    except Exception:  # noqa: BLE001 - keep the UI alive, report in the log
        log.exception("Adjustment Blend failed")
        result = False
    finally:
        cmds.undoInfo(closeChunk=True)

    _report(signal, smart, result)
    return result


def _report(signal, smart, result):
    """Surface a short in-view message; details still go to the Script Editor."""
    mode = "Smart" if smart else "Normal"
    if result:
        msg = f"Adjustment Blend applied  ({signal.title()} / {mode})"
    else:
        msg = f"Adjustment Blend: nothing to process  ({signal.title()} / {mode})"

    log.info(msg)
    try:
        cmds.inViewMessage(statusMessage=msg, position="midCenter",
                           fade=True, fadeStayTime=2000)
    except Exception:  # noqa: BLE001 - inViewMessage is cosmetic
        pass
