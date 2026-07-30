"""
A tiny Maya UI for Adjustment Blend.

One big button plus two toggles — Signal (Scalar / Vector) and Mode
(Normal / Smart) — that call :func:`adjustment_blend.run`. Deliberately thin:
all the real logic lives in the package; this just flips the two flags, wraps
the run in a single undo chunk, and reports the result.

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


def show():
    """Create (or re-show) the Adjustment Blend window."""
    if cmds.window(_WINDOW, exists=True):
        cmds.deleteUI(_WINDOW)

    win = cmds.window(_WINDOW, title="Adjustment Blend", sizeable=False)
    cmds.columnLayout(adjustableColumn=True, rowSpacing=8,
                      columnAttach=("both", 12), width=280)

    cmds.separator(height=6, style="none")

    # The giant button. Command is wired below, once the toggles exist.
    button = cmds.button(
        label="Adjustment Blend",
        height=68,
        annotation="Blend the selected adjustment layer using the options below.",
    )

    cmds.separator(height=6, style="in")

    signal_ctrl = cmds.radioButtonGrp(
        label="Signal",
        labelArray2=["Scalar", "Vector"],
        numberOfRadioButtons=2,
        select=1,
        columnWidth3=(60, 100, 100),
        annotation="Scalar: per-attribute velocity.  "
                   "Vector: one shared speed per channel group.",
    )
    mode_ctrl = cmds.radioButtonGrp(
        label="Mode",
        labelArray2=["Normal", "Smart"],
        numberOfRadioButtons=2,
        select=1,
        columnWidth3=(60, 100, 100),
        annotation="Smart: borrow motion when an attribute has none of its own.",
    )

    cmds.separator(height=6, style="none")

    cmds.button(button, edit=True,
                command=functools.partial(_on_run, signal_ctrl, mode_ctrl))

    cmds.showWindow(win)
    return win


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
