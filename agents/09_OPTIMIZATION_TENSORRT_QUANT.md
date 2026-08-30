# 09 — Native TensorRT Engine + Optional INT8 Quantization

Build a native TensorRT engine from the ONNX export in `08` — this typically beats the ONNX
Runtime TensorRT *execution provider* on latency, since you get direct control over builder flags,
workspace size, and precision, rather than going through ORT's own EP integration layer.

## 1. FP16 engine (do this unconditionally — it's the deployment default)

```bash
trtexec \
  --onnx=results/onnx/winner.onnx \
  --saveEngine=results/trt/winner_fp16.engine \
  --fp16 \
  --memPoolSize=workspace:2048 \
  --verbose > results/trt/build_fp16.log 2>&1
```

`trtexec` prints its own latency/throughput numbers at the end of the build log — record them, but
**also** re-benchmark the engine through your own harness (Python `tensorrt` runtime + the same
CUDA-event timing protocol from `06`) so every row in your final table used identical measurement
methodology. Trusting `trtexec`'s numbers alone for the final report would be inconsistent with
how every other row was measured.

A minimal Python TensorRT inference wrapper (implement as `optimize/trt_runner.py`):
```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: initializes CUDA context

logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
with open("results/trt/winner_fp16.engine", "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()
# allocate input/output device buffers matching the fixed shapes from 08's export, then
# time context.execute_v2(...) in a loop using the same CUDA-event warmup/measure protocol as 06.
```

## 2. INT8 post-training quantization — optional, accuracy-gated

Only adopt INT8 if it **measurably** clears an accuracy bar. Don't ship it just because it's
faster — that's the whole point of measuring rather than assuming.

**Build a calibration set:** ~200 images from FreiHAND's *training* split (not eval — don't
calibrate on your own test set), preprocessed identically to how the model expects input.

```bash
# Using trtexec's built-in calibration flow (simplest path):
trtexec \
  --onnx=results/onnx/winner.onnx \
  --int8 \
  --calib=results/trt/calibration_cache.bin \
  --saveEngine=results/trt/winner_int8.engine \
  --memPoolSize=workspace:2048 \
  --verbose > results/trt/build_int8.log 2>&1
```

If `trtexec`'s calibration flow needs more control than the CLI gives you (custom calibration data
loader), implement `trt.IInt8EntropyCalibrator2` directly in Python instead — check the current
TensorRT Python API docs for the exact interface, since builder APIs shift between TensorRT major
versions.

**The accuracy gate:** re-run the FreiHAND `full` eval subset through the INT8 engine, compute
`pa_mpjpe`, and compare against the FP16 engine's number from step 1.

- If `pa_mpjpe(int8) - pa_mpjpe(fp16) < 2.0mm` (or your own justified threshold — state it if you
  change it): **adopt INT8** as the deployed precision.
- Otherwise: **ship FP16**, and record the INT8 result anyway in the report as "measured, not
  adopted" — a negative result you measured is more valuable than a positive result you assumed.

## 3. Comparison table update

Add `tensorrt-fp16` and `tensorrt-int8` (if built) rows to `results/optimization_table.md`,
completing the full chain:
`pytorch-fp32 → onnxruntime-cuda → onnxruntime-tensorrt-ep → tensorrt-fp16 → tensorrt-int8`

This full chain, with real numbers at every step, **is the deliverable** — it's a direct,
concrete demonstration of "diagnose then optimize" rather than "apply an optimization and hope,"
which is the mindset that mattered in the original LeRobot/KV-cache project this one is modeled
after.

## Definition of Done

- [ ] `results/trt/winner_fp16.engine` exists and is benchmarked through your own harness (not
      just `trtexec`'s self-reported numbers).
- [ ] An INT8 calibration attempt was made; either it was adopted with the accuracy delta stated,
      or explicitly rejected with the measured delta stated — no silent skip.
- [ ] `results/optimization_table.md` shows the full precision/backend chain with consistent
      measurement methodology across every row.
- [ ] Peak VRAM for the TensorRT engine(s) is recorded via `nvidia-smi`, same as every prior stage.
