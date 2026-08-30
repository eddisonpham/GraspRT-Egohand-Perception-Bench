# 00 — Notes: Inductive Biases & Hardware Baseline

## Inductive biases (restated in my own words)

1. **Decoupled detector + reconstruction beats monolithic for egocentric video.** Hands pop in/out
   of egocentric frames at odd angles; a cheap detector that *gates* a heavier reconstruction head
   (WiLoR, HaMeR) means empty/no-hand frames cost almost nothing, whereas a monolithic model pays
   full inference cost every frame regardless of content. Prior: this should win in *effective*
   per-frame latency (and CPU savings) for the target domain.
2. **A model that ships a fast-precision mode is lower-risk.** WiLoR's `--fast` flag (FP16 +
   depth-pruning) is a signal the authors already found a good speed/accuracy operating point —
   the optimization work is partially done, so we inherit a validated trade-off instead of
   discovering one from scratch. Prior: WiLoR-fast should be near the Pareto front.
3. **MANO params beat raw dense-vertex regression for this pipeline.** MANO (θ pose, β shape) is
   low-dimensional, physically regularized, and directly retargetable to a robot hand — the
   downstream consumer of these predictions. Prior: prefer MANO-based models (MobRecon, WiLoR,
   HaMeR) over pure keypoint-only (MediaPipe) when accuracy is comparable.
4. **Keep a deliberately cheap floor reference (MediaPipe).** It's CPU-real-time, no mesh, just
   21 3D keypoints — but if the big transformer models can't clearly beat it on
   speed/accuracy/VRAM combined, that's a legitimate result, not a failure. Prior: MediaPipe is
   the floor, and "bigger is better" is *not* assumed.
5. **One shared interface, one shared harness.** All models behind the same `BaseHandModel` ABC,
   one benchmark harness measuring identical metrics with identical timing methodology — this is
   what makes the comparison apples-to-apples and addition of candidates cheap.

## Hardware baseline (recorded via nvidia-smi, 2026-08-30)

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU (Blackwell, sm_120)
- Driver: 577.13, CUDA Version (driver max): 12.9
- Total VRAM: 8151 MiB
- **Baseline VRAM usage before any workload:** ~1020 MiB (Razer Axon, Brave, Cursor holding memory)
- Effective budget for experiments: ~7.1 GB free at idle; runbook ceiling target stays **≤6.0 GB**
  peak per run for headroom.