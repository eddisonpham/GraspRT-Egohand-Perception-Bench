# 11 — Final Validation, Smoke Tests & Report

Prove the optimized artifact actually works end-to-end, then assemble everything produced by `00`
through `10` into one report a human can read in five minutes and trust.

## End-to-end smoke test (`tests/smoke_test.py`)

Load the deployed artifact (the adopted engine from `09` — FP16 or INT8, whichever was chosen) and:

1. Run it on 5 fresh FreiHAND eval images not used in calibration, assert no crash and
   `joints_3d.shape == (21, 3)` for each.
2. Run it on one full egocentric clip from `04`, frame by frame, assert no crash across the whole
   clip and that per-frame latency doesn't degrade over the run (a growing-latency trend usually
   means a memory leak — VRAM should plateau after a few frames, not climb linearly with frame
   count; check `torch.cuda.memory_allocated()` at frame 5 vs frame 100 if the clip is long enough).
3. Assert peak VRAM during this run is **≤ 6.0GB** (the real ceiling from `00`/`01`, not the
   nominal 7GB) via both the `torch` counter and an `nvidia-smi` poll.
4. Assert mean latency is under whatever target you set in `00` (e.g. <33ms/frame for a 30 FPS
   target) — if it isn't, the honest outcome is documenting that gap, not silently lowering the bar.

```python
def test_smoke():
    engine = load_deployed_engine(...)
    for img in freihand_sample_images(n=5):
        pred = engine.infer(engine.preprocess(img))
        assert pred[0].joints_3d.shape == (21, 3)

    peak_vram_mb = run_on_clip_and_track_vram(engine, "data/egocentric_clips/clip_01.mp4")
    assert peak_vram_mb <= 6000, f"Exceeded VRAM budget: {peak_vram_mb}MB"
```

Run this test to completion and capture its output — don't hand-wave "it should work," run it.

## `results/FINAL_REPORT.md` — assemble, don't re-derive

This file should be built by **pulling real numbers out of the JSON/markdown files already
produced**, not restated from memory. Structure:

1. **Summary** (3–5 sentences): what was compared, what won, what the final deployed latency/VRAM/
   accuracy numbers are.
2. **Model comparison table** — pulled from `results/comparison_table.md` (stage `06`).
3. **Pareto plot** — embed `results/plots/pareto.png`.
4. **Decision** — pulled from `results/decision.md` (stage `07`), including the paragraph on
   whether the inductive biases from `00` held up.
5. **Optimization chain table** — pulled from `results/optimization_table.md` (stages `08`/`09`),
   showing the full `pytorch-fp32 → ... → tensorrt-fp16/int8` progression with real numbers at
   every step, and the INT8 accept/reject decision with its measured accuracy delta.
6. **Advanced optimizations** (stage `10`) — included if attempted, explicitly marked "not
   attempted" if skipped.
7. **Smoke test results** — pass/fail on every assertion above, with actual measured numbers next
   to each target (e.g. "target: <33ms, measured: 24.1ms mean, 29.8ms p95 — PASS").
8. **Known limitations** — be specific: e.g. single dominant hand only, no two-hand interaction, no
   ground truth on the egocentric qualitative clips, MANO-gated models require the license step.
9. **Repro** — the exact environment combo from `01`, all commit hashes from `results/commit_hashes.json`, and the exact commands to re-run the smoke test.

## Definition of Done

- [ ] `tests/smoke_test.py` passes, with its actual printed output saved to
      `results/smoke_test_output.log`.
- [ ] `results/FINAL_REPORT.md` exists, and every number in it is traceable to a specific file
      produced by an earlier stage — no invented or estimated figures.
- [ ] The report explicitly states whether stage `10` was attempted.
- [ ] A reviewer with no prior context on this project could read `FINAL_REPORT.md` alone and
      understand what was built, why, and how well it performs.
