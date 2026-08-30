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
