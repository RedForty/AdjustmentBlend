"""
adjustmentBlendKeys — puts Adjustment Blend's key writing on Maya's undo queue.

Adjustment Blend writes its keys through ``MFnAnimCurve``, which is many times
faster than ``cmds.setKeyframe`` but is invisible to Edit > Undo. Maya only
undoes commands, so this command stands in for the write: it takes the record
of what was changed (an ``MAnimCurveChange``) and replays it backwards on undo
and forwards on redo.

It is not meant to be called by hand. ``adjustment_blend.maya_keys.write_keys``
loads this plug-in, writes the keys and runs the command.
"""

import maya.api.OpenMaya as om

maya_useNewAPI = True


class AdjustmentBlendKeys(om.MPxCommand):
    NAME = "adjustmentBlendKeys"
    VERSION = "1.0"

    def __init__(self):
        super().__init__()
        self._change = None

    @classmethod
    def creator(cls):
        return cls()

    def isUndoable(self):
        return True

    def doIt(self, args):
        # Imported here so a reloaded adjustment_blend is the one that is used.
        from adjustment_blend import maya_keys
        self._change = maya_keys.take_change()

    def undoIt(self):
        self._change.undoIt()

    def redoIt(self):
        self._change.redoIt()


def initializePlugin(plugin):
    om.MFnPlugin(plugin, "KlugTools", AdjustmentBlendKeys.VERSION, "Any").registerCommand(
        AdjustmentBlendKeys.NAME, AdjustmentBlendKeys.creator)


def uninitializePlugin(plugin):
    om.MFnPlugin(plugin).deregisterCommand(AdjustmentBlendKeys.NAME)
