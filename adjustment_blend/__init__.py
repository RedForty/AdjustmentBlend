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

    __all__ = [
        "run",
        "AdjustmentContext",
        "AttributeData",
        "LayerStack",
        "core",
        "__version__",
    ]
else:
    __all__ = ["core", "__version__"]
