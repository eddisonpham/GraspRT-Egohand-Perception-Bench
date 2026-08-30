# 00 — Project Brief

## Goal

Determine, empirically, which off-the-shelf hand pose/mesh estimation model is the best perception
backbone for an egocentric hand-manipulation pipeline, **under a 7GB VRAM budget on an RTX 5060
laptop GPU**, then optimize the winner for fast local deployment using ONNX Runtime and TensorRT.

## Hardware profile (assume this, verify in `01_ENVIRONMENT_SETUP.md`)

- GPU: NVIDIA GeForce RTX 5060 (laptop), Blackwell architecture, CUDA compute capability `sm_120`.
  This is a **new-enough architecture that framework support lags** — PyTorch/TensorRT/ONNX
  Runtime wheels available at any given moment may or may not have `sm_120` kernels compiled in.
  Treat this as a real risk to schedule, not a formality — see `12_TROUBLESHOOTING.md`.
- VRAM: user reports ~7GB usable. Treat 6.0GB as the real ceiling (OS + display compositor +
  browser reserve some amount before your process even starts).
- CPU: plentiful cores available. CPU is a legitimate fallback execution provider for the
  lightest candidate model, and is useful for calibration/data-prep work that doesn't need the GPU.

## Definition of done

1. Four candidate models (see `02`) are each implemented behind one shared interface (see `03`),
   with measured latency, peak VRAM, and accuracy on a common benchmark (see `06`).
2. A written, numbers-backed decision selects one winner (see `07`).
3. The winner is exported to ONNX and to a native TensorRT engine (FP16 at minimum, INT8 evaluated
   and adopted only if it clears an accuracy gate) (see `08`, `09`).
4. A smoke test proves the optimized artifact runs end-to-end on real images within budget, and
   `results/FINAL_REPORT.md` documents the full journey with real numbers (see `11`).

## Non-goals (explicitly out of scope for this project)

- Training a new model from scratch.
- Robot hand retargeting or Isaac Lab policy training — that's the next project once a fast,
  accurate perception backbone exists.
- Multi-hand, two-hand-interaction, or hand-object contact modeling — single dominant hand is fine.
- Squeezing out the absolute best possible mAP — the point is a *defensible, reasoned trade-off*
  under a hard resource constraint, which is the actual skill being demonstrated.

## Inductive biases guiding model selection (state these up front, don't discover them by accident)

These are deliberate, stated priors about what *should* work well given the constraints — the
benchmark in `06` exists to check whether reality agrees with them, not to replace the reasoning.

1. **Decoupled detector + reconstruction beats a monolithic model for egocentric video.** Hands
   enter and leave an egocentric frame constantly, at extreme angles, often partially out of view.
   A cheap, fast detector stage that gates a heavier reconstruction stage (as WiLoR and HaMeR both
   do) should waste less compute on empty/no-hand frames than a model that always runs full cost.
2. **A model that already ships a fast-precision mode is lower-risk than one that doesn't.**
   WiLoR ships a built-in `--fast` flag (FP16 + depth pruning). That is a strong signal the authors
   already validated an accuracy/speed trade-off point, which de-risks your own optimization work.
3. **Parametric (MANO) output is a better fit for this pipeline than raw dense-vertex regression.**
   MANO parameters (pose θ, shape β) are a low-dimensional, directly-retargetable representation —
   valuable for the downstream retargeting project even though it's out of scope here. Prefer
   MANO-based candidates over pure heatmap/dense-mesh regressors when accuracy is comparable.
4. **Always keep one deliberately "cheap" floor reference.** MediaPipe Hand Landmarker produces no
   mesh, just 21 3D keypoints, and is CPU-real-time. If the fancy transformer-based models can't
   clearly beat MediaPipe's speed/accuracy trade-off inside the VRAM budget, that is itself a valid
   and useful finding — don't assume "bigger model wins" going in.
5. **One shared interface, one shared harness.** Every model is wrapped behind the same
   `BaseHandModel` ABC (see `03`) so the benchmark is apples-to-apples and a 5th candidate could be
   added later for the cost of one new wrapper file, not a new benchmark script.

## Definition of Done for this file

- [ ] You can restate, in your own words, why each of the 5 inductive biases above is a reasonable
      prior (not just copy them) — write one sentence per bias into `results/00_notes.md`.
- [ ] You have confirmed the actual GPU model and driver version via `nvidia-smi` and recorded it.
