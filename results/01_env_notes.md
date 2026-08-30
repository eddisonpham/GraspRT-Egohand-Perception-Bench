# 01 — Environment Notes (RTX 5060 / sm_120)

## Working environment combo (2026-08-30)

- OS: WSL2, Ubuntu 26.04 LTS (kernel 6.18.33.2-microsoft-standard-WSL2), Miniconda
- Python: 3.10 (conda env `egohand`)
- **torch 2.11.0+cu128, torchvision 0.26.0+cu128** (stable channel, `--index-url .../whl/cu128`)
  → ships `sm_120` kernels, verified: `torch.cuda.get_device_capability(0) == (12, 0)`,
  no "no kernel image" warning. matmul FP32 + FP16 both OK.
- CUDA runtime used: 12.8 wheel set (cu128), driver supports up to 12.9.
- onnxruntime-gpu 1.23.2 — providers: `TensorrtExecutionProvider`, `CUDAExecutionProvider`,
  `CPUExecutionProvider` (all available)
- tensorrt 11.2.1.2 (pip)
- Other: opencv-python-headless, numpy, scipy, trimesh, matplotlib, tqdm, ultralytics, onnx,
  pillow, onnxconverter-common, psutil

GPU smoke test (`scripts/gpu_smoke_test.py`) passed: `capability: (12, 0)`, matmul OK, peak mem
~210 MB for the smoke test itself.

## Baseline VRAM before any workload

`nvidia-smi` at idle: **1020 MiB / 8151 MiB used** (OS + display compositor + Brave + Cursor).
Per-experiment budget therefore ~7.1 GB at idle; project ceiling stays 6.0 GB peak per run.

## Deviation/notes

- Conda had a new Terms-of-Service gate: ran `conda tos accept` + `--override-channels` once at env
  creation time so env creation wouldn't fail silently.
- Used `opencv-python-headless` in the shared env to avoid GUI OpenCV pulling an extra GNOME/GTK
  stack into WSL headless runs.