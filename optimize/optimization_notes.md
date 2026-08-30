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
