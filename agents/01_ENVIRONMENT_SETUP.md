# 01 — Environment Setup (RTX 5060 / `sm_120` aware)

## Why this file exists

The RTX 50-series (Blackwell, compute capability **12.0 / `sm_120`**) is new enough that many
pre-built PyTorch/TensorFlow/ONNX Runtime wheels do **not** ship kernels for it. The canonical
failure mode is:

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

or a startup warning like:

```
NVIDIA GeForce RTX 5060 with CUDA capability sm_120 is not compatible with the current
PyTorch installation. The current PyTorch install supports CUDA capabilities sm_50 sm_60
sm_70 sm_75 sm_80 sm_86 sm_90.
```

Do not treat this as an edge case to handle later — verify it **first**, before installing any of
the 4 model repos, or you will burn hours debugging a model install when the real problem is the
base framework never touched the GPU in the first place.

## Recommended OS

If on Windows: use **WSL2 with Ubuntu 22.04/24.04**. TensorRT, Triton, and most of these research
repos assume Linux; fighting Windows-native CUDA toolchains is not a good use of time here. If
already on native Linux, skip straight to step 2.

## Steps

1. **Check driver + base CUDA.**
   ```bash
   nvidia-smi
   ```
   Confirm the GPU shown is the RTX 5060 and note the driver version and the "CUDA Version" in the
   top-right of the table (this is the max CUDA runtime the driver supports, not what's installed).
   Blackwell needs a driver new enough to support CUDA 12.8/12.9 — if `nvidia-smi` reports
   something like CUDA 12.4 or lower, update the driver before proceeding.

2. **Create an isolated environment.**
   ```bash
   conda create -n egohand python=3.10 -y
   conda activate egohand
   ```

3. **Install PyTorch — stable first, nightly as fallback.** Try the current stable `cu128` wheel:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```
   Run the smoke test in step 6 immediately. If you hit the "no kernel image" error, that means
   `sm_120` support has not landed in stable yet — fall back to the nightly channel, which tends to
   carry Blackwell kernels earlier:
   ```bash
   pip uninstall torch torchvision -y
   pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu129
   ```
   Record which of the two worked in `results/01_env_notes.md` — this matters for reproducibility
   and for anyone reading your project who has the same GPU.

4. **Install ONNX Runtime GPU + TensorRT.**
   ```bash
   pip install onnxruntime-gpu
   pip install tensorrt
   ```
   TensorRT 10.x+ is required for Blackwell support. If `pip install tensorrt` gives you an old
   version, install NVIDIA's TensorRT tarball/deb for your CUDA version directly from the NVIDIA
   Developer site instead, matched to the CUDA toolkit version your PyTorch build actually uses.

5. **Install shared CV/tooling deps.**
   ```bash
   pip install opencv-python numpy scipy trimesh matplotlib tqdm ultralytics onnx pillow
   ```
   (`ultralytics` is a dependency of WiLoR's detector stage — install now so `05` goes smoothly.)

6. **GPU smoke test.** Save and run this as `scripts/gpu_smoke_test.py`:
   ```python
   import torch
   print("torch:", torch.__version__)
   print("cuda available:", torch.cuda.is_available())
   if torch.cuda.is_available():
       print("device:", torch.cuda.get_device_name(0))
       print("capability:", torch.cuda.get_device_capability(0))  # expect (12, 0)
       a = torch.randn(4096, 4096, device="cuda")
       b = torch.randn(4096, 4096, device="cuda")
       c = a @ b
       torch.cuda.synchronize()
       print("matmul OK, result sample:", c[0, 0].item())
       print("peak mem MB:", torch.cuda.max_memory_allocated() / 1e6)
   ```
   This must print `capability: (12, 0)` and complete the matmul with no error.

7. **Baseline VRAM check.** Before any model is loaded, check what the OS/driver is already
   holding:
   ```bash
   nvidia-smi --query-gpu=memory.used,memory.total --format=csv
   ```
   Record this number. Your real per-experiment budget is `6000MB - this_baseline`, not 6000MB.

## Definition of Done

- [ ] `nvidia-smi` shows the RTX 5060 and a driver supporting CUDA ≥ 12.8.
- [ ] `gpu_smoke_test.py` runs with no error and prints capability `(12, 0)`.
- [ ] `onnxruntime-gpu` imports and `onnxruntime.get_available_providers()` includes
      `CUDAExecutionProvider` (and `TensorrtExecutionProvider` if installed via the full TensorRT
      package rather than the pip-only subset).
- [ ] Baseline VRAM usage before any workload is recorded in `results/01_env_notes.md`.
- [ ] The exact working combination of `torch`/`torchvision`/CUDA wheel index is written down —
      this environment is what every later benchmark number is conditioned on.
