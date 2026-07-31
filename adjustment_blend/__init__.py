"""
Adjustment Blend — velocity-matched keyframe interpolation for Maya.

Drop an additive animation layer with just a start and an end key to offset
some existing motion, then let this tool fill in the in-betweens. Instead of a
flat linear blend, the offset is distributed across the segment in proportion
to how fast the *underlying* animation is moving — so contacts stay planted,
holds stay held, and the adjustment rides along with the motion that's already
there.

Based on Dan Low's GDC talk on adjustment blending.

Typical use, from inside Maya, after selecting your controls and the
adjustment layer::

    import adjustment_blend
    adjustment_blend.run(smart=True)

Or drive it explicitly (headless / pipeline / tools), supplying the layer
context yourself instead of reading it from the Anim Layer editor::

    adjustment_blend.run(
        adjustment_layer="AnimLayer1",
        layers_below=["BaseAnimation"],
        objects=["pCube1"],
    )

The algorithm itself lives in :mod:`adjustment_blend.core` and has no Maya
dependency, so it can be unit tested and reused anywhere.
"""

from __future__ import annotations

__version__ = "0.1.0"

# The pure core is always importable (no Maya required) so tests and tooling
# can use it directly.
from . import core  # noqa: F401


def reload_all():
    """Reload every submodule so code edits take effect without restarting Maya.

    The builtin ``reload()`` is shallow — it re-runs a package's ``__init__`` but
    not its already-imported submodules — so edits to ``ui.py``, ``pipeline.py``,
    etc. never show up. Call this during development instead. It reloads in
    dependency order and refreshes this package's own bindings in place, so your
    existing import keeps working (no reassignment needed)::

        from klugTools import adjustment_blend
        adjustment_blend.reload_all()
        adjustment_blend.show_ui()

    Returns the (reloaded) package module.
    """
    import importlib
    import sys

    pkg = __name__
    # Leaves first, package last, so each reload sees fresh dependencies.
    for sub in ("core", "vector_core", "maya_layers", "maya_scene", "pipeline", "ui"):
        module = sys.modules.get(f"{pkg}.{sub}")
        if module is not None:
            importlib.reload(module)
    return importlib.reload(sys.modules[pkg])


def build_stamp():
    """Return a ``'v<version> · <date time>'`` string for the running build.

    The timestamp is the most recent modification time across the package's
    source files — i.e. when this build was last written or deployed — so it
    tracks reality with no manual version-date bookkeeping. Handy in the UI
    tooltip to confirm which build (and how fresh) is actually loaded.
    """
    import glob
    import os
    import time

    sources = glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))
    try:
        newest = max(os.path.getmtime(f) for f in sources)
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(newest))
    except (ValueError, OSError):
        when = "unknown"
    return f"v{__version__} · {when}"

# The scene-facing API only loads inside Maya. Guarding the import keeps
# ``import adjustment_blend`` (and ``adjustment_blend.core``) working in plain
# CPython for unit tests and CI.
try:
    import maya.cmds  # noqa: F401
    _HAS_MAYA = True
except ImportError:
    _HAS_MAYA = False

if _HAS_MAYA:
    from .pipeline import run, AdjustmentContext, AttributeData  # noqa: F401
    from .maya_layers import LayerStack  # noqa: F401
    from .ui import show as show_ui  # noqa: F401

    __all__ = [
        "run",
        "show_ui",
        "reload_all",
        "build_stamp",
        "AdjustmentContext",
        "AttributeData",
        "LayerStack",
        "core",
        "__version__",
    ]
else:
    __all__ = ["core", "reload_all", "build_stamp", "__version__"]
