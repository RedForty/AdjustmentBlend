# Adjustment Blend review items

Recorded 2026-11-14 from the code review of that day. I re-checked every item against the code of that day in headless Maya 2025. An item marked "from reading" has no measurement behind it.

Paths are inside `adjustment_blend/` unless they name another folder.

Severity:

- **High.** Wrong keys in ordinary use.
- **Medium.** A wrong or misleading result in a less common setup, or time lost on every run.
- **Low.** Cleanup. No wrong result today.

Size is my guess at the work: small is under an hour, medium is an afternoon.

## Open items

| ID | Severity | Item | Size |
|---|---|---|---|
| AB-10 | Low | Two signals, Scalar and Vector, do the same job | decision |
| AB-12 | Low | Code that nothing reads | small |
| AB-13 | Low | Three small things | small |
| AB-15 | Low | `scripts/adjustmentBlend.py` is a legacy copy of the tool | small |

### AB-10 Low. Two signals, Scalar and Vector, do the same job

Where: `pipeline.py:254-367`, `core.py:131-158`.

Scalar with Smart needs `apply_smart_fallbacks`, `_motion_below`, `_find_axis_fallback`, `_find_channel_fallback`, `get_other_axis` and `get_other_channel`, about 140 lines, to let a still axis follow a moving one. Vector's shared speed per channel does that with no fallback code.

Decision for you, from use: if Vector feels right on real shots, delete Scalar and the Signal toggle.

### AB-12 Low. Code that nothing reads

- `calculate_adjustment_values`, `pipeline.py:209`. A wrapper around one call.
- The `skipped` dict, `pipeline.py:151-200`. It feeds one debug log line.
- `scale_speed`, `vector_core.py:117`. A one-line alias of `translation_speed`.
- `distribute_by_speed`, `vector_core.py:172`. Since the two loops were merged it only calls `core.distribute_adjustment`.
- `normalize_values`, `core.py:105`. Nothing in the package calls it now. `tests/test_core.py` tests it.
- `normalize_attr_name`, `SHORT_TO_LONG`, `ROTATION_ATTRS` and `SCALE_ATTRS`, `core.py:31-44`. Only the old `LayerStack` used them. `tests/test_core.py` still tests the function, so remove those two tests with it.

### AB-13 Low. Three small things

- `ui.py:40-41`. `show()` deletes the workspace control every time, so Maya forgets where you docked it. From reading.
- `vector_core.py:1-19`. The docstring says experimental and not wired in. It is wired in.
- `pipeline.py:451-458`. The dry-run warning and the INFO log put every attribute name on one line. At 100 controls that is 900 names. Print a count.

### AB-15 Low. `scripts/adjustmentBlend.py` is a legacy copy of the tool

2,163 lines, the single-file version from before the package. No script under `KlugTools/scripts` imports it. I did not check shelves or hotkeys. Delete it once those are checked.

## Closed

| Item | Outcome |
|---|---|
| A failed run read as "nothing to process", was AB-06 | Fixed. The panel says FAILED and points at the Script Editor, and `run()` raises when the keys cannot be written instead of returning False. 3 scene checks in the new `ui` group |
| `layers_below` did nothing, was AB-05 | Removed in 0.2.0. `run(adjustment_layer, objects)`. The tool no longer needs the Anim Layer editor: the stack order comes from `animLayer -children`, which I checked against the order Maya composites in, with moved and nested layers |
| With two layers selected, name order picked the adjustment layer, was AB-04 | Fixed. `maya_scene.discover_adjustment_layer` takes the topmost selected layer in the stack. 3 scene checks |
| Key times from every control and attribute were merged, was AB-02 | Fixed. Every curve is blended between its own keys. 4 scene checks. Two controls with the same name under different groups now work too; before, the run refused them |
| A third of a run went to listing animated attributes, was AB-07 | Fixed. `maya_scene.layer_curves` walks the layer once. 100 controls by 120 frames: 0.10 s in Scalar and 0.14 s in Vector, from 0.25 s |
| A key on a fractional frame left a stray key, was AB-08 | Fixed. Only whole frames between a curve's first and last key are written, and a key between frames keeps its value. 1 scene check |
| Smart fallback wrote borrowed speed into shared data, was AB-11 | Fixed. A fallback always reads the sibling's own motion from the `LayerStack`, so the processing order cannot matter |
| Float noise on a still channel defeated the blend, was AB-03 | Fixed. `core.is_still` treats steps up to `STILL_TOLERANCE`, 1e-6 per frame, as no motion. 3 scene checks on baked keys with noise and 3 pytest checks |
| A key between the first and last gave a pop or overwrote a key, was AB-01 | Fixed in `core.segment_weights`. A span no longer counts the motion that arrived at its first key, and a span with no motion below spreads its change evenly. 4 scene checks and 5 pytest checks |
| Two copies of the distribution loop, was AB-09 | Fixed. `distribute_by_speed` calls `core.distribute_adjustment` |
| `LayerStack` did not compute the layers below | Fixed. Rewritten, 21 checks in `adjustment_blend/tests/test_layers.py` |
| Cost grew with the square of the number of controls | Fixed. One shared `LayerStack` per run |
| `run()` from a script gave one undo step per key | Fixed. `adjustmentBlendKeys` makes a run one step, 20 checks in `adjustment_blend/tests/test_keys.py` |
| No scene tests | Fixed. 80 checks, `mayapy adjustment_blend/tests/run.py`. The review said there were no tests at all. That was wrong: `tests/` already had 39 pytest checks for the pure core, now 46 |
| Dead code in the old `maya_layers.py` | Gone with the rewrite |
| Move the undo chunk into `run()` | Not needed after the plug-in |
| Replace `maya_layers.py` with a `getAttr(time=)` sampler | Withdrawn. The spike measured it about 250 times slower per sample |

## What is left

Every High and Medium item is closed. The rest is cleanup:

1. AB-12 and AB-13. Small.
2. AB-15. Delete the legacy file once shelves and hotkeys are checked.
3. AB-10 once you have an opinion on Vector.
