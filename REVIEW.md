# Adjustment Blend review items

Recorded 2026-11-14 from the code review of that day. I re-checked every open item against the current code in headless Maya 2025. An item marked "from reading" has no measurement behind it.

Paths are inside `adjustment_blend/` unless they name another folder.

Severity:

- **High.** Wrong keys in ordinary use.
- **Medium.** A wrong or misleading result in a less common setup, or time lost on every run.
- **Low.** Cleanup. No wrong result today.

Size is my guess at the work: small is under an hour, medium is an afternoon.

## Open items

| ID | Severity | Item | Size |
|---|---|---|---|
| AB-02 | High | Key times from every control and attribute are merged | medium |
| AB-03 | High | Float noise on a still channel defeats the blend | small |
| AB-04 | Medium | With two layers selected, name order picks the adjustment layer | small |
| AB-05 | Medium | `layers_below` does nothing | small |
| AB-06 | Medium | A failed run reads as "nothing to process" | small |
| AB-07 | Medium | A third of a run goes to listing animated attributes | medium |
| AB-08 | Low | A key on a fractional frame leaves a stray key | small |
| AB-10 | Low | Two signals, Scalar and Vector, do the same job | decision |
| AB-11 | Low | Smart fallback writes borrowed speed into shared data | small |
| AB-12 | Low | Fields and code that nothing reads | small |
| AB-13 | Low | Four small things | small |
| AB-15 | Low | `scripts/adjustmentBlend.py` is a legacy copy of the tool | small |

### AB-02 High. Key times from every control and attribute are merged

Where: `maya_scene.py:144-165`, `pipeline.py:122-137`.

The run collects the key times of every curve into one set and cuts every curve at all of them. A curve gets spans it has no keys for. Each of those spans is blended by itself, so the offset no longer arrives in step with the motion over the artist's own span.

Measured:

- Control `a` is keyed at 0 and 40. Blended alone it holds from frame 30 to 40. Blended together with `b`, keyed at 0 and 30, `a` is 5.06 at frame 30 and drifts to 6.0 by frame 40. It moves 0.94 during a hold, in all four modes. Before the fix for the keys-in-the-middle bug this was a one-frame pop.
- One control with translateX keyed at 0 and 40 and rotateY keyed at 0, 25 and 40. translateX moves by up to 0.24 of its 6.0 offset compared with blending translateX alone.
- `b` gets keys out to frame 40, fifteen frames past its last key.

The existing check `blend, two controls that hold at different times` does not catch this because both controls have the same key times.

Fix: build the spans per curve from that curve's own keys. AB-07 is the same change seen from the speed side.

### AB-03 High. Float noise on a still channel defeats the blend

Where: `core.py:51-57`.

`is_equal` calls a channel still only when every sample is bit-for-bit the same. Baked and mocap keys on a still channel rarely are.

Measured on translateY, still at 2.0, with a 0 to 4.0 offset:

| Keys below | Mode | Result |
|---|---|---|
| exact | Normal | left sparse, as intended |
| exact | Smart | follows translateX, moves 0.000 in the 10 to 20 hold |
| 1e-9 noise | Normal | follows the noise, moves 0.909 in the hold |
| 1e-9 noise | Smart | same as Normal, never borrows |

Fix: call a channel still when its total travel over the range is under a threshold. I'd start at 1e-6 in UI units, but the right number depends on your data, so try it on a real baked shot. Add a check with noisy keys to `adjustment_blend/tests/test_blend.py`.

### AB-04 Medium. With two layers selected, name order picks the adjustment layer

Where: `maya_scene.py:38-44`, `maya_scene.py:75-78`.

`get_selected_animlayers` returns layers in `cmds.ls` order and the code takes the last one. Measured: the stack is BaseAnimation, Zeta, Alpha, Mid from the bottom. With Alpha and Zeta selected it returns `['Alpha', 'Zeta']`, so Zeta gets the keys. Zeta is the lower of the two.

Fix: decide the rule first. The topmost selected layer is my suggestion. Then read the order from the stack. Depends on what you decide in AB-05.

### AB-05 Medium. `layers_below` does nothing

Where: `pipeline.py:93-95`, `pipeline.py:120`, `maya_scene.py:47-93`, `maya_scene.py:123-141`.

Measured: three different `layers_below` lists gave identical keys. The reader takes what Maya shows below the layer straight from the blend nodes and never looks at the list.

What the argument still does:

- The run aborts when the list is empty.
- `run()` with no arguments needs the Anim Layer editor widget `AnimLayerTabanimLayerEditor` to exist, only to build this list. From reading: without the widget `cmds.treeView` raises.
- Locked layers are dropped from the list, which changes nothing.

Decision for you: remove it or give it a meaning. I'd remove it. That deletes `filter_layers_by_objects`, the tree view query and most of `discover_layers`, and the tool stops depending on the editor being open.

### AB-06 Medium. A failed run reads as "nothing to process"

Where: `ui.py:130-139`, `ui.py:149`. From reading.

The panel catches every exception, logs the traceback and sets the result to False. The in-view message then says nothing to process. A failed key write returns False the same way. The artist cannot tell "nothing to blend" from "the tool broke".

Fix: a separate message for a failure, pointing at the Script Editor.

### AB-07 Medium. A third of a run goes to listing animated attributes

Where: `maya_scene.py:96-120`, called from `maya_scene.py:153` and `pipeline.py:175`.

Profile of a full run, 100 controls by 120 frames, 0.37 s:

| Part | Time |
|---|---|
| `get_animated_attributes`, twice per control | 0.12 s |
| `distribute_adjustment` | 0.07 s |
| Writing 108,000 keys | 0.04 s |
| Reading the motion below | 0.04 s |
| Everything else | 0.10 s |

Fix: walk the adjustment layer's own curves once instead of asking each control for its animated plugs. That also removes the OpenMaya 1.0 code and the repeated membership queries, and it is the loop AB-02 needs. I expect about 0.1 s back at 100 controls. That is an estimate, so benchmark it.

`distribute_adjustment` can also lose time: it calls `list(calculation_range).index` for every span and `map_from_to` once per key, 108,000 calls here.

### AB-08 Low. A key on a fractional frame leaves a stray key

Where: `core.py:101-111`, `core.py:212`, `pipeline.py:251-254`.

Measured with the first key at frame 0.5: the run writes a key at frame 0, before the artist's first key, holding 0.0174 of a 6.0 offset. Every other frame is within that of the right value. I rated this higher in the review. Measured, it is small.

Fix: write only the whole frames strictly between a span's two keys. Do it with AB-02.

### AB-10 Low. Two signals, Scalar and Vector, do the same job

Where: `pipeline.py:269-383`, `core.py:114-137`.

Scalar with Smart needs `apply_smart_fallbacks`, `_find_axis_fallback`, `_find_channel_fallback`, `get_other_axis` and `get_other_channel`, about 140 lines, to let a still axis follow a moving one. Vector's shared speed per channel does that with no fallback code.

Decision for you, from use: if Vector feels right on real shots, delete Scalar and the Signal toggle. AB-11 goes with it.

### AB-11 Low. Smart fallback writes borrowed speed into shared data

Where: `pipeline.py:299`, read back at `pipeline.py:329` and `pipeline.py:370`. From reading.

When an attribute borrows a speed, the code stores it as if it were the attribute's own. A later attribute can then borrow the borrowed speed. I tried to build a case where the keys change because of it and could not. With three axes per channel, the borrowed speed is the one the direct rule would pick anyway. So it is fragile but not wrong today.

Fix: keep the borrowed speed in a separate field. Not needed if AB-10 removes Scalar.

### AB-12 Low. Fields and code that nothing reads

- `AdjustmentContext.layers_to_process` and `.adjustment_keys`, `pipeline.py:43` and `:45`.
- `AttributeData.velocity_source`, `pipeline.py:59`, set at `:205` and `:300`.
- `calculate_adjustment_values`, `pipeline.py:225`. A wrapper around one call.
- The `skipped` dict, `pipeline.py:162-216`. It feeds one debug log line.
- `scale_speed`, `vector_core.py:117`. A one-line alias of `translation_speed`.
- `distribute_by_speed`, `vector_core.py:172`. Since the two loops were merged it only calls `core.distribute_adjustment`.
- `normalize_values`, `core.py:88`. Nothing in the package calls it now. `tests/test_core.py` tests it.
- `isinstance(layers_below, list)`, `maya_scene.py:80`. Always true.
- `normalize_attr_name`, `SHORT_TO_LONG`, `ROTATION_ATTRS` and `SCALE_ATTRS`, `core.py:31-45`. Only the old `LayerStack` used them. `tests/test_core.py:25-31` still tests the function, so remove those two tests with it.

### AB-13 Low. Four small things

- `maya_scene.py:69` compares against the literal `"BaseAnimation"`. Use the root layer the function already queried.
- `ui.py:40-41`. `show()` deletes the workspace control every time, so Maya forgets where you docked it. From reading.
- `vector_core.py:1-19`. The docstring says experimental and not wired in. It is wired in.
- `pipeline.py:469-476`. The dry-run warning and the INFO log put every attribute name on one line. At 100 controls that is 900 names. Print a count.

### AB-15 Low. `scripts/adjustmentBlend.py` is a legacy copy of the tool

2,163 lines, the single-file version from before the package. No script under `KlugTools/scripts` imports it. I did not check shelves or hotkeys. Delete it once those are checked.

## Closed

| Item | Outcome |
|---|---|
| A key between the first and last gave a pop or overwrote a key, was AB-01 | Fixed in `core.segment_weights`. A span no longer counts the motion that arrived at its first key, and a span with no motion below spreads its change evenly. 4 scene checks and 5 pytest checks |
| Two copies of the distribution loop, was AB-09 | Fixed. `distribute_by_speed` calls `core.distribute_adjustment` |
| `LayerStack` did not compute the layers below | Fixed. Rewritten, 21 checks in `adjustment_blend/tests/test_layers.py` |
| Cost grew with the square of the number of controls | Fixed. One shared `LayerStack` per run |
| `run()` from a script gave one undo step per key | Fixed. `adjustmentBlendKeys` makes a run one step, 20 checks in `adjustment_blend/tests/test_keys.py` |
| No scene tests | Fixed. 65 checks, `mayapy adjustment_blend/tests/run.py`. The review said there were no tests at all. That was wrong: `tests/` already had 39 pytest checks for the pure core, now 43 |
| Dead code in the old `maya_layers.py` | Gone with the rewrite |
| Move the undo chunk into `run()` | Not needed after the plug-in |
| Replace `maya_layers.py` with a `getAttr(time=)` sampler | Withdrawn. The spike measured it about 250 times slower per sample |

## Order I'd do them in

1. AB-03. Small, and it decides whether the tool works on baked data at all.
2. AB-07 with AB-02 and AB-08. One change: walk the layer's curves, spans per curve.
3. AB-05, then AB-04. Both need you to say what layer selection should mean.
4. AB-06, AB-12, AB-13.
5. AB-10 once you have an opinion on Vector.
