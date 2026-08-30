# 08 — ONNX Export & ONNX Runtime Optimization

Export the winner from `07` (and, for comparison, the runner-up if time allows) to ONNX, then
benchmark it under ONNX Runtime's CUDA and TensorRT execution providers against the raw PyTorch
baseline already measured in `06`. This isolates "how much does graph-level optimization alone
buy you" before native TensorRT engine-building in `09`.

## 1. Export to ONNX

```python
# optimize/export_onnx.py
import torch

model.eval()
dummy_input = torch.zeros(1, 3, 224, 224, device="cuda")  # match the model's actual input size/crop
torch.onnx.export(
    model,
    dummy_input,
    "results/onnx/winner.onnx",
    export_params=True,
    opset_version=17,
    input_names=["image"],
    output_names=["mano_pose", "mano_shape", "joints_3d"],  # match your model's real outputs
    dynamo=True,   # prefer the newer dynamo-based exporter if your PyTorch version supports it
)
```

**Fix the batch size and input resolution for export.** Egocentric-frame variable hand counts and
dynamic crops are handled by your existing detector/preprocessing stage *outside* the exported
graph — the exported graph itself should be a fixed-shape, single-crop forward pass. Don't fight
dynamic axes here; it adds real complexity for no benefit at batch size 1.

**Common failure mode:** custom ops (grid_sample, ROI-align-style cropping, some attention
variants) used inside WiLoR/HaMeR/MobRecon may not export cleanly to a given opset. If export
fails or produces a graph with `PythonOp`/unsupported-op warnings:
- Try `dynamo=True` with `report=True` to get a diagnostic report pinpointing the failing op.
- If a specific op genuinely won't export, consider exporting only the backbone + regression head,
  and re-implementing the small amount of pre/post-processing (bbox crop, NMS if any) as plain
  Python/NumPy code that runs outside the ONNX graph, calling the ONNX session for the expensive
  part only. This is a legitimate, common pattern — not a failure.

## 2. Verify the export

```python
import onnx
m = onnx.load("results/onnx/winner.onnx")
onnx.checker.check_model(m)
```
Then run one sample through both the original PyTorch model and the ONNX Runtime session and
confirm outputs match within a small numerical tolerance (e.g. `np.allclose(..., atol=1e-3)`) —
catching a silently-wrong export here is much cheaper than catching it via a confusing accuracy
regression in step 4.

## 3. Benchmark under ONNX Runtime execution providers

```python
import onnxruntime as ort

providers_to_try = [
    ["CPUExecutionProvider"],
    ["CUDAExecutionProvider", "CPUExecutionProvider"],
    ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],  # if installed
]
for providers in providers_to_try:
    sess = ort.InferenceSession("results/onnx/winner.onnx", providers=providers)
    # warm up 20 iters, then time 200 iters exactly as in 06 (reuse the same profiling utilities)
```

Note: the first call to a session using `TensorrtExecutionProvider` triggers TensorRT engine
building under the hood, which is slow (can be seconds to a minute) — **exclude this from your
warmup-then-measure protocol by warming up ≥20 iterations before timing**, same rule as always.

## 4. Re-check accuracy after export

Re-run the FreiHAND `dev` subset (not necessarily full, to save time) through the ONNX Runtime
session and recompute `pa_mpjpe`. It should match the PyTorch baseline from `06` closely (small
floating-point differences are expected and fine; a large jump means something in the export or
pre/post-processing split is wrong — go back to step 1's verification, don't proceed past this).

## Output

Append new rows to `results/comparison_table.md` (or a new `results/optimization_table.md`) for:
`pytorch-fp32` (from `06`, for reference) → `onnxruntime-cpu` → `onnxruntime-cuda` →
`onnxruntime-tensorrt-ep`, each with latency/VRAM/accuracy in the same schema as `06`.

## Definition of Done

- [ ] `results/onnx/winner.onnx` exists, passes `onnx.checker.check_model`, and its outputs match
      the PyTorch model within tolerance on at least 5 sample images.
- [ ] Latency/VRAM benchmarked under CPU, CUDA, and (if available) TensorRT execution providers,
      using the same warmup/timing protocol as `06`.
- [ ] Accuracy re-verified on the FreiHAND dev subset post-export and confirmed to match baseline.
- [ ] Results appended to the comparison table with clear labels distinguishing each EP.
