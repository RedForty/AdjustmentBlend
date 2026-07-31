# Adjustment Blend

**Velocity-matched keyframe interpolation for Maya animation layers.**

Drop an additive animation layer with just a *start* and an *end* key to offset
some existing motion, then let Adjustment Blend fill in the in-betweens. Instead
of a flat linear blend, the offset is distributed across the segment in
proportion to how fast the *underlying* animation is moving — so contacts stay
planted, holds stay held, and the adjustment rides along with the motion that's
already there.

Based on [Dan Low's GDC talk](https://www.youtube.com/watch?v=eeWBlMJHR14) on
adjustment blending.

---

## Requirements

- Autodesk Maya 2022 or newer (Python 3).

## Install

Adjustment Blend is a normal Python package. Pick whichever fits your pipeline:

- **Drop-in (artists / TDs):** copy the `adjustment_blend/` folder into a folder
  on your Maya `PYTHONPATH` (e.g. `~/maya/scripts`).
- **Editable / headless / CI:** `pip install -e .` into the interpreter you want
  (including `mayapy`).

## Usage

### Interactive (the everyday workflow)

1. Animate your objects.
2. Add **one** animation layer to offset your animation. Key only the **first**
   and **last** frames on that layer.
3. Select the objects that share that adjustment layer (and select the layer in
   the Anim Layer editor).
4. Run:

```python
import adjustment_blend
adjustment_blend.run(smart=True)
```

With no arguments, `run()` reads the adjustment layer and the stack beneath it
from the Anim Layer editor and operates on your current selection.

### UI

For a click-to-run panel with the two modes side by side (handy for A/B'ing
Scalar vs Vector on a shot):

```python
import adjustment_blend
adjustment_blend.show_ui()
```

A compact, **dockable** panel: one **Adjustment Blend** button over two toggles —
**Signal** (Scalar / Vector) and **Mode** (Normal / Smart). Dock it anywhere and
Maya remembers the spot; each run is a single undo step; and your toggle choices
persist across scenes and sessions.

### Explicit / headless (pipeline, batch, tests)

Pass the layer context yourself to skip all UI queries:

```python
import adjustment_blend

adjustment_blend.run(
    adjustment_layer="AnimLayer1",
    layers_below=["BaseAnimation"],
    objects=["pCube1"],
    smart=True,
)
```

### Options

| Argument           | Default      | Meaning                                                                 |
| ------------------ | ------------ | ----------------------------------------------------------------------- |
| `adjustment_layer` | *discovered* | Target layer to densify. Read from the UI when omitted.                 |
| `layers_below`     | *discovered* | Ordered layers to composite motion from. Read from the UI when omitted. |
| `objects`          | *selection*  | Objects to process. Falls back to selection, then to layer members.     |
| `signal`           | `"scalar"`   | Motion signal. `"scalar"` = per-attribute velocity. `"vector"` (experimental) = one shared speed per channel group. |
| `smart`            | `False`      | Enable fallbacks when an attribute has no motion of its own (see below). |
| `apply`            | `True`       | Set `False` for a dry run that computes but writes nothing.             |

## How it works

Maya's layer blend is linear: it spreads your start→end offset evenly over time,
which slides feet and drifts contacts because it ignores what the base animation
is doing. Adjustment Blend instead samples the **velocity (speed) graph** of the
layers below the adjustment layer and hands out the offset *in proportion to that
motion*. Fast frames absorb most of the change; held frames barely move; the
artist's keyed endpoints are always honored exactly.

The pipeline runs in four stages:

1. **`build_context`** — resolve layers/objects, collect the adjustment keys.
2. **`collect_attribute_data`** — read composite motion below via `LayerStack`.
3. **`distribute_adjustment`** — the velocity-weighted distribution (pure math).
4. **`apply_keyframes`** — write the densified curve back to Maya.

`smart=True` adds a fallback: when an attribute's own underlying motion is flat,
it borrows motion from the sibling axis (or channel) whose net displacement best
matches the adjustment — e.g. a horizontal `rotateY` offset can ride the stride
already present in `translateZ`.

### Experimental: `signal="vector"`

The scalar signal looks at each attribute's own velocity, which goes flat when
you introduce motion on an axis the base animation doesn't use (the reason
`smart` exists). The **vector** signal instead reduces each channel group to a
single speed — translation speed for `translate`, geodesic **angular speed** for
`rotate` (built from all three Euler axes plus the rotation order), scale speed
for `scale`. Because all three axes share one signal, a flat axis automatically
rides whatever its siblings are doing below, so the sibling-axis fallback is no
longer needed. In this mode `smart` governs only the last-resort *cross-channel*
borrow (e.g. a fully static rotation group riding translation). Flat segments
distribute linearly rather than collapsing.

The algorithm lives in `adjustment_blend/vector_core.py` (pure, unit tested).
It's opt-in while it gets real-scene mileage; the scalar path is unchanged.

## Project layout

```
adjustment_blend/
├── core.py          # pure algorithm — no Maya, fully unit tested
├── maya_layers.py   # LayerStack: walks Maya's blend-node chains
├── maya_scene.py    # scene I/O + layer discovery (the only UI dependency)
└── pipeline.py      # orchestration + the public run() entry point
tests/
└── test_core.py     # algorithm tests (run with plain pytest, no Maya)
```

The algorithm lives entirely in `core.py` with **no Maya dependency**, so it can
be unit tested and reused from anywhere. The Maya-facing modules only read values
out of the scene and write keyframes back.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The test suite covers the pure core and needs no Maya install.

## Known limitations

- Segments are evaluated independently; an adjustment spanning a region where
  the base animation is flat in one segment but not another can produce uneven
  results (smart mode mitigates this).
- Override layers are treated as additive.

## License

See [LICENSE.md](LICENSE.md).
