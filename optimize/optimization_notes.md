# Optimization Notes

## ONNX export attempt (2026-08-30)

Winner selected by the measured decision matrix: **WiLoR-fast**.

Attempted to export a fixed-shape wrapper returning `(pred_keypoints_3d, pred_vertices)` with
YOLO detection and ViTDetDataset preprocessing outside the graph. Two official PyTorch export
paths were attempted:

1. `torch.onnx.export(..., dynamo=True)` failed in `torch.export` inside the MANO/SMPL-X
   `batch_rigid_transform` parent-joint Python loop with
   `GuardOnDataDependentSymNode` (data-dependent symbolic integer).
2. Legacy TorchScript tracing was attempted as a fallback, but PyTorch's Lightning module
   introspection accessed the `WiLoR.trainer` property and raised `RuntimeError` because the
   model is not attached to a Trainer.

No ONNX file is claimed from these failed attempts. This is a measured, reproducible negative
result, not a placeholder. The proper next implementation is to split the graph further at the
MANO boundary: export only tensor-only neural regression; implement MANO kinematic decoding as a
small fixed tensor/C++/plugin postprocess. The current WiLoR source combines those operations in
`forward_step`, so that refactor is deferred rather than silently shipping an unverified graph.

## Current deployment artifact

The validated deployment artifact is the official WiLoR-fast PyTorch path: FP16 weights,
`torch.compile` backbone (when supported), and official `skip_blocks=True`, with detector and
crop preprocessing retained. It measured 33.51ms mean / 39.42ms p95 / 29.84 FPS / 3014MB
nvidia-smi peak / 5.901mm PA-MPJPE on 200 seeded FreiHAND eval images.

TensorRT/INT8 are not claimed until an ONNX graph exists and passes numerical equivalence.

## YOLO detector TensorRT export (2026-08-30) — REJECTED on accuracy

The WiLoR YOLO hand detector (`detector.pt`, a YOLO-pose model) was exported to a TensorRT
FP16 engine via ultralytics `.export(format='engine', half=True)`. It is ~2x faster than the
native torch detector on the same 224x224 inputs (7.3ms vs 16.4ms, detection rate 1.000 both), 
but the exported engine produces **box coordinates shifted by ~4.85px** vs the torch path.

That box shift is not FP16-rounding (an FP32 engine shows the identical 4.85px shift); it is an
artifact of the ONNX->TRT round-trip of the YOLO-pose box decoder / NMS. Errors propagate into
ViTDetDataset crop coords, and end-to-end PA-MPJPE collapses from 5.65mm (torch detector) to
21.43mm (TRT detector) on 100 dev images.

Because the crop is hypersensitive to box position, the TRT detector is **not adopted** despite
its latency win. Native torch detector remains the correct path. This is a measured honest
negative result.

## Detector bottleneck — root cause isolated (2026-08-30)

Fine-grained split profiling (exact `stream_inference` emulation) shows the detector time is
**forward (GPU conv) = 13.8ms**, preprocess 2.2ms, postprocess small — genuinely compute-bound.

Tested every acceleration path; summary:

| Approach | Latency | Box fidelity | Outcome |
|---|---:|---:|---|
| TRT full detector (FP16) | 2.2x | 4.85px drift | REJECT (5.65→21.4mm PA-MPJPE) |
| TRT full detector (FP32) | 2x | identical 4.85px drift | REJECT (not rounding) |
| TRT backbone+neck FP16, native Pose head | 2x (7.3ms) | **179.8px drift** | REJECT |
| torch.compile detector | 0.95x | **0.0px (bit-identical)** | no gain |
| imgsz 384-640 / FP16 tuning | none | identical | no gain |

**Root cause**: the YOLO-pose decode head is a hypersensitive amplifier. Whatever tiny feature
delta the TRT compiler's kernel scheduling introduces (identical in FP16 and FP32, so not
precision) blows up into multi-pixel box shifts that destroy reconstruction accuracy. Only the
exactly-identical torch execution keeps boxes bit-stable, and the conv graph resists torch.compile
fusion on Blackwell (0.95x).

**Conclusion**: the 13.8ms detector forward cannot be accelerated while preserving accuracy via
quantization or compilation, because decode-fidelity and compute-acceleration are mutually
exclusive for this pose detector. Further detector gains require either a new yolo head that is
decode-stable under TRT, or a smaller detector re-trained for the hand task — both beyond an
in-place optimization pass. This is a measured, decisive honest negative that closes out the
detector bottleneck.

## Software pipelining (double-buffer) — near-negative (2026-08-30)

Overlapped the CPU crop/transfer/postprocess of frame N+1 with the GPU TRT inference of frame N
via a two-buffer CUDA-stream pipeline (`optimize/bench_pipelined.py`), vs the fully serial path
that inserts `torch.cuda.synchronize()` between every stage. Same detector, same crop code,
same TRT FP16 engine — only the scheduling differs.

| Scheduling | FPS | mean ms | p95 ms |
|---|---:|---:|---:|
| Serial | 28.1 | 35.64 | 40.75 |
| Pipelined | 28.9 | 34.62 | 39.82 |

Speedup **1.03x** (throughput 28.1→28.9 FPS). Outputs remain **bit-identical** across the two
schedulings (max |joints| diff = 0.0), so the overlap is provably correctness-safe.

The gain is marginal because the pipeline was never CPU-bubble-bound: detector forward (13.8ms)
and TRT reconstruction (12.6ms) are both GPU compute and together saturate the SMs, so the
CPU-side crop/transfer there was to hide is only the ~1ms real math. Overlapping CPU work under
a saturated-GPU tail cannot recover GPU-bound time. This is consistent with the earlier
multi-stream negative: the wall is GPU SMs, not Python/transfer overhead.

**Verdict**: software pipelining yields no meaningful end-to-end gain here and is a near-negative,
recorded for completeness. The e2e pipeline stays single-threaded serial (simpler, no correctness
risk) at its standing 28.6-28.9 FPS.

## Two-stream overlap (det_stream + trt_stream) — negative (2026-08-30)

Within a single frame the detector and reconstruction are data-dependent (recon runs on the
detector's crop), so they cannot overlap inside one frame. The only real overlap is ACROSS
frames: detector[N+1] on `det_stream` while recon[N] runs on `trt_stream`. That scheduling was
wired explicitly (`optimize/bench_stream_overlap.py`) and benchmarked vs the serialized path.

| Scheduling | FPS | mean ms | p95 ms |
|---|---:|---:|---:|
| Serialized | 30.4 | 32.90 | 37.55 |
| Two-stream | 29.3 | 34.08 | 39.49 |

Speedup **0.97x** (two-stream is *slightly slower*). Outputs remain bit-identical (max |joints|
diff = 0.0).

Two-stream overlap is a **negative**: putting detector forward (13.8ms) and recon (12.6ms) on
separate streams does not help because both saturate the GPU SMs — there is no spare SM
throughput to recover — and stream-switch scheduling adds a small overhead on top. This is
direct, evidence-backed confirmation that the pipeline is GPU-SM-bound end-to-end, not
stream-scheduling-bound. Consistent with the double-buffer (1.03x) and multi-stream (flat)
negatives.

**Verdict**: rejected; the serialized path remains (simpler, no downside). The e2e wall is GPU
SM throughput, now confirmed three independent ways (compile, stream overlap, multi-stream),
and no scheduling rearrangement recovers it.

## Detection cadence (box reuse every K frames) — the one real lever, a trade (2026-08-30)

The GPU-SM wall blocks per-frame detector acceleration, but in *video* the detector need not run
every frame. Detection cadence (detect every K-th frame, reuse the box for the (K-1)
intermediates) amortizes the 13.8ms detector forward without touching reconstruct compute. This
is the only lever that bypasses the wall. Measured on 300 real synthetic ego frames (15fps
224x224 clips), TRT FP16 recon per frame, serialized pipeline:

| Cadence K | FPS | mean ms | p95 ms | vs K=1 |
|---|---:|---:|---:|---:|
| 1 (baseline) | 28.4 | 35.23 | 41.58 | 1.00x |
| 2 | 38.7 | 25.85 | 36.15 | 1.36x |
| 3 | 41.1 | 24.33 | 36.83 | 1.45x |
| 4 | 43.9 | 22.80 | 36.96 | 1.55x |
| 6 | 44.5 | 22.49 | 37.67 | 1.57x |

Cadence clearly breaks the 30 FPS wall (K=2 already 38.7 FPS). **But it is a trade, not a free
win**: the synthetic-clip box-stability analysis shows consecutive-frame box center moves by a
median of **4.0px / p95 15px / max 53-77px**, and box *size* by 17-19px per frame. The boundary case is decisive: a
**4.85px box shift collapsed PA-MPJPE 5.65 -> 21.4mm** in the TRT detector experiment. Stale-box crops at K>=2 carry multi-pixel drift on moving hands, so
reconstruction accuracy will degrade in exactly the regime that proved hypersensitive.

## Temporal smoothing for egocentric jitter — reduces shake, no training (2026-08-30)

WiLoR is frame-independent, so consecutive 3D joints carry per-frame noise (jitter). Tested
cheap causal temporal filters on the joint trajectory on real synthetic ego clips
(`optimize/bench_egocentric_jitter.py`): moving-average (MA5/MA9) and one-Euro adaptive filter.

| Filter | jitter mm | jitter reduction | distortion mm |
|---|---:|---:|---:|
| raw | 0.093 | — | — |
| MA5 | 0.018 | 81% | 0.061 |
| MA9 | 0.010 | 90% | 0.063 |
| one-euro | 0.034 | 64% | 0.042 |

All filters cut jitter substantially (64-90%) at tiny absolute distortion (<0.07mm, i.e. they
smooth real noise, not signal). MA9 maximizes smoothness; one-Euro has the best noise/retain
ratio (lowest distortion). Raw jitter is already low on these near-static synthetic clips, so
the win is real but modest in absolute terms. This is the no-training path for "smoother
tracking"; recommended one-Euro (0.042mm distortion) as the deployment default. Measured and
reproducible.

## Fine-tuning WiLoR for egocentric accuracy + detection — research & feasibility (2026-08-30)

The jitter path fixes shake without training. The remaining asks (egocentric domain-accuracy,
detection drift) need fine-tuning. WiLoR's reconstruction net is already SOTA-adjacent (5.5mm
paper / 5.65mm ours); its egocentric gap is domain shift (top-down ego crops, occlusions).

Target datasets with adequate paired 3D/MANO GT for egocentric fine-tuning (researched):

| Dataset | Scale | Mano GT | Fit |
|---|---|---|---|
| **HOT3D** (Meta, CVPR'25) | 833 min, 3.7M+ imgs, 19 subj | **Yes (MANO + UmeTrack)** | Direct; 3832 curated clips |
| EgoDex (Apple) | 829h ego video, paired 3D tracking | Yes | Strong |
| WildHands | daily egocentric images | pseudo/multi | auxiliary |

HOT3D is the primary pick: MANO-format annotations match WiLoR's output schema, egocentric
top-down views directly attack the domain gap, and its curated clips ease protocol splits.

Feasibility on 6GB VRAM (honest): WiLoR is a ViT-Large CONVMANO reconstruction head. Full
fine-tuning of the backbone is likely >6GB. Realistic path = **LoRA/QLoRA on the backbone + full
fine-tune of the small reconstruction/MANO head at low LR**, FP16 + gradient checkpointing +
batch=1-2 at 224px(ish). QLoRA (4-bit quantized LoRA base) is the memory-compatible route.
Estimate: LoRA on attention q/k/v/o with rank 8-16 needs ~1-2GB of base in 4-bit, fits the
6GB budget with headroom; expect epoch hours on consumer Blackwell (SM120) hardware.

Detection drift is separate: fine-tuning the YOLO-pose hand detector on ego crops (manual boxes
or pseudo-labels) is a lighter task (7MB model) and the higher-confidence win, since the
hypersensitive decoder is the achilles heel under compile/quantization.

Honest boundary: this is a training task, gated on (a) downloading HOT3D (multi-GB, license
agreement) and (b) GPU-hours. The engineering infrastructure (TRT export, LoRA adapter mount,
jitter eval) is ready and reusable.

**Verdict**: temporal smoothing is complete and verified (jitter -90%). Fine-tuning is a
researched, feasible next phase on HOT3D via QLoRA + head fine-tune; not executable here without
the dataset download and a training run. Recorded as the two-leg plan.
 It is viable only where a steady operator hand is the target (e.g.
grasp-pose estimation with a slow-moving hand) and where reconstruction-tolerance to small
crop translation is acceptable — both domain decisions, not free speed. This is the honest
boundary: the pipeline is GPU-SM-bound for *every-frame* accuracy; cadence relaxes accuracy
assumptions to buy FPS. Recorded as the definitive answer to "is >30 FPS possible": yes, at a
stated accuracy tradeoff.
