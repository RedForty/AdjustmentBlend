"""
A tiny, dockable Maya UI for Adjustment Blend.

One big button plus two toggles — Signal (Scalar / Vector) and Mode
(Normal / Smart) — that call :func:`adjustment_blend.run`. Deliberately thin:
all the real logic lives in the package; this just flips the two flags, wraps
the run in a single undo chunk, and reports the result.

The panel is a ``workspaceControl``, so it docks anywhere in the Maya UI and
Maya remembers where. The two toggles persist to ``optionVar`` so a user's
choice is remembered across scenes and sessions.

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

_WORKSPACE = "adjustmentBlendWorkspaceControl"

# Persistent preferences (1 = first option, 2 = second option).
_OPT_SIGNAL = "adjustmentBlendSignal"   # 1 = Scalar, 2 = Vector
_OPT_MODE = "adjustmentBlendMode"       # 1 = Normal, 2 = Smart


def show():
    """Create (or re-show) the dockable Adjustment Blend panel."""
    if cmds.workspaceControl(_WORKSPACE, exists=True):
        cmds.deleteUI(_WORKSPACE)

    # The uiScript lets Maya (re)build the panel's contents on demand — when it
    # is docked, floated, restored from a saved layout, or rebuilt on startup.
    # Build the call from this module's real name so it resolves whatever the
    # install path is (klugTools.adjustment_blend.ui, adjustment_blend.ui, ...).
    ui_script = f"import {__name__} as _ab_ui; _ab_ui._build_ui()"

    cmds.workspaceControl(
        _WORKSPACE,
        label="Adjustment Blend",
        uiScript=ui_script,
        retain=False,
        floating=True,
    )
    return _WORKSPACE


def _build_ui():
    """Populate the panel. Called by Maya via the ``uiScript``.

    Maya sets the current parent to the workspaceControl before calling this, so
    the layout below lands inside the dockable frame.
    """
    from . import build_stamp

    cmds.columnLayout(adjustableColumn=True, rowSpacing=4,
                      columnAttach=("both", 8), width=140)

    cmds.separator(height=3, style="none")

    # Tooltip carries the build version + date so you can confirm what's loaded.
    button = cmds.button(
        label="Adjustment Blend",
        height=36,
        annotation=f"Adjustment Blend  {build_stamp()}\n"
                   f"Blend the selected adjustment layer using the options below.",
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
