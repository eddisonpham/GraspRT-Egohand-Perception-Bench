# 06 — Benchmark Harness

One harness, run once per model (per-model subprocess/env as noted in `05`), writing one JSON file
each to `results/raw/<model_name>.json`. `aggregate.py` then builds the comparison table and plot.

## Metrics to capture, and exactly how

### Latency
- Warm up 20 iterations first (`model.warmup(20)`) — **do not skip this**. First-call CUDA context
  init, cuDNN/cuBLAS autotuning, and lazy kernel compilation are real one-time costs that do not
  represent steady-state performance, and will visibly distort your numbers if included.
- Time 200 iterations on a single representative image (or the "dev" FreiHAND subset, looped).
- Use CUDA events for GPU-side models, not wall-clock `time.perf_counter()` alone — Python-side
  timers include host-side dispatch overhead that CUDA events don't:
  ```python
  start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
  start.record()
  preds = model.infer(batch)
  end.record()
  torch.cuda.synchronize()
  latency_ms = start.elapsed_time(end)
  ```
  For MediaPipe (CPU-only), plain `time.perf_counter()` around `infer()` is correct and sufficient.
- Report: mean, median, p95, std, across the 200 timed iterations. **p95, not just mean** — a
  control-loop-adjacent use case cares about tail latency, not just average throughput.
- Derive throughput: `fps = 1000.0 / mean_latency_ms`.

### Peak VRAM
- Use both of these, and report both — they can legitimately disagree, and the gap is informative:
  ```python
  torch.cuda.reset_peak_memory_stats()
  # ... run the timed loop ...
  torch_peak_mb = torch.cuda.max_memory_allocated() / 1e6
  ```
  ```bash
  # background poll during the same run, e.g. every 50ms in a separate thread/process:
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
  ```
  `torch`'s counter tracks allocator-level usage; `nvidia-smi` reflects true driver-level usage
  including CUDA context overhead, fragmentation, and anything outside PyTorch's allocator (e.g.
  TensorRT's own memory pool later in `09`). For MediaPipe, skip this section (CPU model).

### Accuracy
- Run each model over the FreiHAND `dev` subset first to catch bugs cheaply, then over `full` for
  numbers that go in the final report.
- Compute, per image: `pa_mpjpe` (always), plus `pa_mpvpe`, `f_score@5mm`, `f_score@15mm` for
  models with `mesh_verts` (B, C, D — not A).
- Aggregate as mean ± std across the subset.

### Model size
- On-disk checkpoint size in MB — cheap, informative, and a reasonable proxy for "how annoying is
  this to ship/version."

## `results/raw/<model>.json` schema

```json
{
  "model": "wilor",
  "variant": "default",
  "commit": "<git sha from 05>",
  "device": "cuda:0 (RTX 5060)",
  "latency_ms": {"mean": 0.0, "median": 0.0, "p95": 0.0, "std": 0.0},
  "fps": 0.0,
  "vram_mb": {"torch_peak": 0.0, "nvidia_smi_peak": 0.0},
  "accuracy": {
    "subset": "full",
    "pa_mpjpe_mm": 0.0,
    "pa_mpvpe_mm": null,
    "f_score_5mm": null,
    "f_score_15mm": null
  },
  "model_size_mb": 0.0,
  "n_eval_images": 0,
  "notes": ""
}
```

Run WiLoR twice (`variant: "default"` and `variant: "fast"`) — this is one of your core comparison
rows, not an afterthought, since the fast mode is a built-in optimization you get almost for free.

## `benchmark/aggregate.py`

Build:
1. A single markdown comparison table (one row per model/variant) from all `results/raw/*.json`
   files, saved to `results/comparison_table.md`.
2. A Pareto scatter plot: x = `pa_mpjpe_mm` (lower better), y = `latency_ms.mean` (lower better),
   marker size ∝ `vram_mb.nvidia_smi_peak`. Save to `results/plots/pareto.png`. This single plot is
   the fastest way for a human reviewer to see the whole trade-off space at a glance.

## Qualitative pass on egocentric clips

For each model, run frame-by-frame inference on the 2–3 egocentric clips from `04` and save an
annotated output video (skeleton or mesh overlay) to `results/qualitative/<model>_clip01.mp4`.
No scoring here — this is for the write-up's "does this actually look right for the target domain"
section, and for catching a model that scores fine on FreiHAND but visibly breaks on egocentric-
style motion blur / close range / partial hand visibility.

## Definition of Done

- [ ] `results/raw/` has one JSON per model/variant (5 files: mediapipe, mobrecon, wilor,
      wilor-fast, hamer — hamer may be "reduced protocol" per `05` if it didn't fit).
- [ ] `results/comparison_table.md` and `results/plots/pareto.png` exist and are readable.
- [ ] Qualitative annotated clips exist for at least the 3 GPU-based models (A/B/C, and D if it ran).
- [ ] Every number in the JSON files came from an actual run in this session — no placeholders.
