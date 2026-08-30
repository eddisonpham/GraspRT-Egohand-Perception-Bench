# 12 — Troubleshooting

Known failure modes, roughly in the order you're likely to hit them. Append to this file whenever
you solve something not already listed here — it's meant to grow.

## `RuntimeError: CUDA error: no kernel image is available for execution on the device`

**Cause:** your PyTorch/TensorFlow/ONNX Runtime build has no compiled kernels for `sm_120`
(Blackwell). This is a real, common issue for RTX 50-series cards, not a misconfiguration on your
part — many stable framework releases lag new architectures by months.

**Fix, in order of preference:**
1. Check for a newer stable release — by the time you're reading this, official `sm_120` support
   may have landed in the stable channel. Check the current PyTorch release notes.
2. If stable still doesn't cover it, use the nightly wheel index (`.../whl/nightly/cu129` or
   whatever the current newest CUDA-version nightly is) — nightlies typically carry new-arch
   support well before stable.
3. As a last resort, build PyTorch from source with `TORCH_CUDA_ARCH_LIST="12.0"` — slow (expect
   30–90+ minutes) but guaranteed correct if steps 1–2 haven't caught up yet.
4. Verify the fix with `python -c "import torch; print(torch.cuda.get_device_capability(0))"` —
   must print `(12, 0)` with no warning before the fix is considered done.

## VRAM OOM at seemingly small batch sizes / models

Your real budget is tighter than "7GB" — see `00`/`01`. Mitigations, in order of how much they
cost you in complexity:
1. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` as an environment variable — reduces
   fragmentation-driven OOMs without changing any code.
2. Fully exit the Python process between benchmarking different models rather than trying to
   `del model; torch.cuda.empty_cache()` and reuse the same process — CUDA context + allocator
   fragmentation across very different model architectures is not worth fighting.
3. Close anything else touching the GPU before benchmarking (browser hardware acceleration, other
   apps) — on a 7GB laptop card this is not optional, it's a meaningful fraction of your budget.
4. Run inference in `torch.no_grad()` / `model.eval()` always — obvious, but double-check every
   wrapper actually does this; a stray gradient graph is a very easy way to blow a small budget.
5. Try `.half()` (FP16) inference before reducing resolution or batch further — cheaper accuracy
   cost than resolution reduction, in general, for these model classes.
6. If a TensorRT engine build itself OOMs (this can happen during the *build* step, independent of
   inference-time OOM), reduce `--memPoolSize=workspace:<N>` in `trtexec` — the builder's search
   over kernel/tactic candidates can temporarily use more memory than the final engine needs to run.

## MANO registration blocking B/C/D

This is a manual, click-through license agreement at https://mano.is.tue.mpg.de/ — **the agent
cannot complete this on the user's behalf.** If you reach this in `05` and it hasn't been done:
stop, clearly tell the user this specific step is needed, give them the URL, and wait. Don't try
to find the model file from an unofficial mirror — respect the license gate as designed.

## ONNX export fails on a custom op (grid_sample, ROI-crop, custom attention variants)

See `08` step 1's failure-mode note — try `dynamo=True, report=True` for a diagnostic first. If a
specific op is fundamentally unsupported at your opset, split the graph: export only the
expensive, static-shape backbone/head to ONNX, and keep small, dynamic pre/post-processing steps
(cropping, NMS) as plain Python outside the ONNX graph. This is standard practice, not a workaround
to be ashamed of.

## `onnxruntime-gpu` imports but `CUDAExecutionProvider` isn't in `get_available_providers()`

Almost always a CUDA/cuDNN version mismatch between what `onnxruntime-gpu` was built against and
what's installed. Check the `onnxruntime-gpu` release notes for the exact CUDA/cuDNN version it
expects, and make sure that matches (or is a compatible minor version with) what `01` installed —
don't assume the newest CUDA toolkit is always compatible with the newest ONNX Runtime release,
these are pinned relationships.

## A model's accuracy looks far worse than the paper reports

Almost always a preprocessing mismatch (crop, resize, normalization mean/std, or color order
BGR vs RGB), not a bug in the model itself. Before suspecting the model or your metric code:
1. Visualize the exact tensor going into `infer()` as an image — does it look like a sane,
   correctly-cropped, correctly-normalized hand crop?
2. Compare against the original repo's own `demo.py`/inference script line-by-line for
   preprocessing — this is the single most common source of silent accuracy bugs in this project.

## Latency numbers are noisy / inconsistent between runs

- Confirm warmup actually ran (20+ iterations) before timing began.
- Confirm no other GPU-using process is running concurrently (check `nvidia-smi` for other
  processes on the same GPU during your benchmark run).
- On a laptop, check for thermal throttling on long benchmark runs — if `nvidia-smi
  --query-gpu=clocks.sm,temperature.gpu --format=csv` shows falling clocks over a long run, your
  later iterations are running slower for a hardware reason, not a software one; either note this
  in the report or add a cooldown between long runs.
