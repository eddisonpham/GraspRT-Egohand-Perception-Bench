# Architecture

A hand-reconstruction benchmark and optimization pipeline targeting a constrained GPU (6 GB VRAM). Four candidate models are measured against FreiHAND ground truth, scored on a weighted latency/accuracy/VRAM/complexity matrix, and the winner is taken through ONNX export and TensorRT quantization.

## Data flow

```
FreiHAND images ──► FreiHandLoader ──► BaseHandModel.preprocess ──► BaseHandModel.infer
       │                                                                      │
       └── GT joints/verts (m) ───────► metrics (PA-MPJPE, PA-MPVPE, F-score) ◄─┘
                                              │
                                              ▼
                                    results/raw/*.json ──► aggregate.py ──► comparison_table.md
                                                                                   decision.md
                                                                                   plots/pareto.png
```

The optimization path branches off `infer()` for the winning model only:

```
WiLoR-fast ──► export_onnx (trace function, bypass Lightning) ──► *.onnx (FP16)
                                                                  │
                          ┌───────────────────────────────────────┤
                          ▼                                       ▼
                   validate_onnx (Δ vs PyTorch)           build_trt_engine (TRT 10.16)
                          │                                       │
                          ▼                                       ▼
                   bench_onnx_latency                     bench_trt + bench_trt_gate
                   (CPU/CUDA EP)                          (latency + accuracy gate vs GT)
                                                                  │
                                                                  ▼
                                                       build_trt_int8 (PTQ, REJECTED)
```

## Module responsibilities

**`common/`** — shared, dependency-free primitives. No GPU, no model code.
- `interface.py` — `BaseHandModel` ABC (`load`/`preprocess`/`infer`/`device`) and the `HandPrediction` output dataclass (21 joints in meters, optional MANO params + 778-vertex mesh). Every wrapper returns this contract.
- `metrics.py` — pure NumPy. Umeyama similarity alignment (rotation + translation + uniform scale, no reflection). `pa_mpjpe` is the headline accuracy metric; all errors in millimeters.
- `profiling.py` — the single timing protocol used across every backend: CUDA events for GPU, `perf_counter` for CPU, `NvidiaSmiMonitor` thread for driver-level VRAM, `write_raw` for the JSON schema.

**`data/`** — dataset access.
- `freihand/loader.py` — `FreiHandLoader` resolves root via `FREIHAND_ROOT` → `~/egohand_data/freihand` → repo fallback. Loads all GT JSON once at construction; images are 224×224 BGR. `__getitem__` returns `(image, gt_joints, K)` in meters.
- `freihand/build_subsets.py` — seeded `dev` (200) and `full` (3960) index files.
- `synthesize_egocentric.py` — qualitative-only synthetic clips derived from FreiHAND; not ground truth.

**`models/`** — one `BaseHandModel` subclass per candidate. Each is isolated: weights and source live outside the repo (`WILOR_ROOT`, `MOBRECON_ROOT`, `HAMER_ROOT`).
- `mediapipe_wrapper.py` — CPU floor reference. 21 landmarks, no mesh.
- `mobrecon_wrapper.py` — DenseStack mesh recon. Spiral ordering is **provisional** (BFS approximation; OpenMesh half-edge winding not reproducible without a compiler).
- `wilor_wrapper.py` — official WiLoR. `variant="fast"` applies FP16 + `torch.compile` + skip-blocks. Wraps the YOLO detector → ViTDet crop → MANO reconstruction pipeline.
- `hamer_wrapper.py` — integration boundary only. Fails explicitly when Detectron2/ViTPose/MANO are absent; emits no synthetic result.

**`benchmark/`** — orchestration.
- `run_benchmark.py` — one model per process. Loads model, warms up, times `infer()` with the shared protocol, measures accuracy over the subset, writes `results/raw/<model>.json`.
- `aggregate.py` — reads all raw JSON, normalizes metrics (min-max), applies fixed weights (latency 0.35, PA-MPJPE 0.35, VRAM 0.20, complexity 0.10), ranks, writes comparison table + Pareto plot + decision matrix. Absent VRAM (CPU models) gets the median of present norms — neutral, not rewarded or penalized.

**`optimize/`** — the winning model's path to deployment.
- `export_onnx.py` — traces a plain function (bypasses Lightning `trainer` introspection), unwraps `torch.compile`, freezes `requires_grad_`. Outputs a 1.2 GB FP16 graph with external constants.
- `validate_onnx.py` — numeric diff vs PyTorch (CPU + CUDA EP).
- `bench_onnx_latency.py` — ORT latency with `pct`/`summarize` helpers (nearest-rank percentiles).
- `build_trt_engine.py` — native TRT 10.16 (cu12) FP16 engine build.
- `bench_trt.py` / `bench_trt_gate.py` — TRT latency + accuracy gate vs GT (pass threshold: PA-MPJPE within 0.5 mm of FP16 baseline).
- `build_trt_int8.py` — INT8 entropy PTQ. Honest negative: REJECTED (+3.89 mm; MANO LBS chain amplifies quantization error).
- `bench_breakdown.py` — per-stage latency (preprocess/detector/reconstruction/postprocess).
- `finetune_wilor_lora.py` — LoRA fine-tune of the WiLoR backbone on 3D-joint GT. Per-epoch val MPJPE monitoring, `ReduceLROnPlateau`, patience early stop with best-state snapshot+restore. Writes the best adapter + reported `val_history`.
- `merge_wilor_lora.py` — folds a saved LoRA adapter into the backbone (`merge_and_unload`) so inference has zero per-layer LoRA cost; re-verifies the held-out MPJPE gain survives the merge.
- `bench_egocentric_jitter.py` — temporal-smoothing study (MA5/MA9, one-Euro) on synthetic ego clips; reports jitter reduction and distortion.
- `bench_box_cadence.py` — detector-cadence (box reuse every K frames) throughput + box-stability analysis on ego clips.
- `bench_pipelined.py` / `bench_stream_overlap.py` — software-pipelining / two-stream scheduling negatives (GPU-SM bound).

**`scripts/`** — `gpu_smoke_test.py` validates VRAM ceiling and leak-free iteration.

**`tests/`** — 74 tests (69 pass, 5 skip on Windows). Unit tests for metrics/profiling/interface; black-box known-answer audits for metrics, loader, aggregate, optimize IO; white-box invariant tests for Umeyama, percentiles, loader round-trip.

## Design decisions

**One model per process.** `run_benchmark.py` imports lazily and resets the allocator between load and timing. CUDA context init and cuDNN autotune are excluded via `warmup()` before any timed iteration.

**Shared timing protocol.** `common/profiling.time_infer` is the only implementation of warmup + iteration + percentile logic. Every backend (PyTorch, ORT, TRT) uses it so latency rows are comparable.

**GT units are meters, errors are millimeters.** `metrics.py` multiplies residuals by 1000. Callers pass meters; the `f_score` unit bug (comparing aligned-mm predictions against unaligned-meter GT) was caught and fixed during the audit.

**No synthetic accuracy.** If a model cannot run, it emits no row — `hamer_wrapper` raises, `wilor_wrapper` returns `[]` on detector miss. The aggregate skips JSON without `latency_ms` + `accuracy`.

**Honest negatives recorded.** MobRecon's spiral ordering, INT8 PTQ, HaMeR's environment block, and the scheduling negatives (pipelining, two-stream) are all documented with the measured numbers, not silently dropped.

**Fine-tuning is merge-first.** The LoRA adapter is folded into the backbone weights before deployment, so the fine-tune improves held-out accuracy (7–13% joint error) with zero change to inference latency or VRAM. Training is fp32 (FP16 autocast hangs fp32-LoRA on this setup); it early-stops on a monitored validation set to avoid overfitting. The egocentric-domain gain is the untested upside gated on a license'd dataset (HOT3D).

## External dependencies

| Asset | Location | License |
|---|---|---|
| WiLoR weights | `$WILOR_ROOT/pretrained_models/` | CC-BY-NC-ND |
| MANO model | `$WILOR_ROOT/mano_data/` | research-only |
| MobRecon checkpoint | `$MOBRECON_ROOT/downloads/` | research-only |
| FreiHAND | `~/egohand_data/freihand` | CC-BY-NC |
| MediaPipe task | `models/assets/` | Apache-2.0 |

The repo code is MIT; downloaded assets carry their own stricter terms.
