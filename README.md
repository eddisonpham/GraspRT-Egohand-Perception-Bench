# EgoHand-Bench

**A controlled benchmark and TensorRT optimization pipeline for hand pose / mesh models on a 7 GB-VRAM laptop GPU (RTX 5060, Blackwell `sm_120`).**

Four candidates (MediaPipe, MobRecon, WiLoR, HaMeR) are benchmarked on a common seeded FreiHAND dev subset under a fixed decision matrix; the winner is then pushed through ONNX → native TensorRT FP16 with an accuracy-gated INT8 step. Every reported number is measured, not assumed.

---

## Headline results

### Model comparison (200-image FreiHAND dev subset)

| Model | Variant | Mean ms | p95 ms | FPS | PA-MPJPE mm | VRAM MB | Score |
|---|---|---:|---:|---:|---:|---:|---:|
| **WiLoR** | **fast** | **33.51** | **39.42** | **29.84** | **5.901** | **3,014** | **0.6942** |
| MediaPipe | default (CPU) | 34.35 | 46.50 | 29.11 | 15.883 | — | 0.6364 |
| MobRecon | DenseStack | 22.71 | 33.78 | 44.03 | 24.77* | 1,292 | 0.6000 |
| WiLoR | default | 69.53 | 77.27 | 14.38 | 5.742 | 4,235 | 0.4000 |

\* MobRecon accuracy is **provisional** — see Limitations.

### Optimization of the winner (WiLoR-fast reconstruction graph)

| Backend | Mean ms | p95 ms | FPS | Accuracy gate |
|---|---:|---:|---:|---|
| **TensorRT FP16 engine** | **13.72** | **18.76** | **72.90** | 5.544 mm vs GT ✅ |
| Native PyTorch (FP16) | 24.01 | 31.55 | 41.65 | 5.9 mm vs GT (baseline) |
| ORT CUDA EP | 52.11 | 84.30 | 19.19 | 0.66 mm vs PyTorch |
| ORT CPU EP | 1209 | 1599 | 0.83 | 0.66 mm vs PyTorch |
| TRT INT8 PTQ | 10.30 | 13.31 | 97.05 | **9.438 mm vs GT — REJECTED** (+3.89 mm) |

→ **TensorRT FP16 is the deployed engine**: 1.75× faster than native, 3.8× faster than ORT-CUDA, inside the 6 GB VRAM budget, accuracy gate passed. INT8 was honestly rejected by the accuracy gate.

Full report: [`results/FINAL_REPORT.md`](results/FINAL_REPORT.md).

---

## What this is

- A **linear runbook** (`agents/00`→`12`) that takes you from environment setup → dataset prep → model integration → benchmarking → decision → ONNX/TensorRT optimization → final report.
- A **shared, fair harness**: one `BaseHandModel` interface, one FreiHAND loader, one CUDA-event latency + VRAM profiler, one Procrustes-aligned PA-MPJPE/PA-MPVPE/F-score metric set.
- **Measure, don't assume**: no model is ranked without real numbers; blocked candidates (HaMeR) and provisional ones (MobRecon) are documented as such, never faked.

---

## Setup

### 1. System
- **GPU:** NVIDIA RTX 5060 Laptop (Blackwell `sm_120`), 8 GB VRAM. Works on any CUDA-12.x Blackwell/Ada/Ampere GPU.
- **OS:** Windows 11 with **WSL2 Ubuntu 26.04** for GPU/TensorRT work; native Windows conda only for MediaPipe (it needs `libGLESv2`, which is absent from a sudo-less WSL).
- **Driver:** ≥ 575.x (CUDA 12.9 capable).

### 2. Conda env (WSL)
```bash
# Miniconda in WSL, then:
conda create -n egohand python=3.10 -y
conda activate egohand

# PyTorch with sm_120 / CUDA 12.8 support (Blackwell):
pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# Project deps (note the TensorRT pin — see "Known issues"):
pip install -r requirements.txt
```

### 3. Assets (manual, license-restricted)
Set environment variables (e.g. in `~/.bashrc`):
```bash
export WILOR_ROOT=$HOME/src/WiLoR            # git clone https://github.com/rolpotamias/WiLoR
export MOBRECON_ROOT=$HOME/src/HandMesh      # git clone https://github.com/SeanChenxy/HandMesh
export FREIHAND_ROOT=$HOME/egohand_data/freihand
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1    # official WiLoR ckpt + Ultralytics loader
```
- **FreiHAND** (evaluation set with annotations, 724 MB): see `agents/04_DATASET_PREP.md`.
- **MANO** (`MANO_RIGHT.pkl`): register at https://mano.is.tue.mpg.de/ and place under each model's expected `mano_data/` path.
- **WiLoR checkpoints**: follow `~/src/WiLoR/README.md`.

### Known issues
- **chumpy on numpy≥2**: `chumpy` imports deleted numpy aliases. Patch after install:
  ```bash
  python -c "p=__import__('chumpy').__file__.replace('/__init__.py',''); s=open(p).read(); open(p,'w').write(s.replace('from numpy import bool, int, float, complex, object, unicode, str, nan, inf','from numpy import nan, inf'))"
  ```
- **TensorRT version**: `onnxruntime-gpu 1.23.2`'s TensorRT EP links `libnvinfer.so.10`. The default `tensorrt` wheel is a CUDA-13 build (`libnvinfer.so.11`) and fails with `cudaError 35`. Pin `tensorrt-cu12==10.16.1.11` (matched to the cu128 nvidia runtime libs).
- **detectron2 (HaMeR)**: no prebuilt wheel for torch 2.11 / `sm_120`, needs `nvcc` to compile. HaMeR is therefore environment-blocked in this setup.

---

## Usage

### Benchmark all models
```bash
# WiLoR-fast on the dev subset (one process, CUDA-event timing)
python benchmark/run_benchmark.py --model wilor --variant fast --iters 200 --subset dev
python benchmark/run_benchmark.py --model mediapipe --iters 200 --subset dev   # Windows
python benchmark/run_benchmark.py --model mobrecon --iters 200 --subset dev

# Aggregate into the comparison table + decision matrix + Pareto plot
python benchmark/aggregate.py
```

### Optimize the winner
```bash
# 1. Export WiLoR-fast reconstruction graph to ONNX (detector/crop stay outside)
python optimize/export_onnx.py --variant fast --out results/onnx/wilor-fast.onnx

# 2. Numeric validation vs native PyTorch
python optimize/validate_onnx.py --onnx results/onnx/wilor-fast.onnx

# 3. ORT backend latency (native wins; documented negative for ORT-CUDA)
python optimize/bench_onnx_latency.py --iters 200

# 4. Build + benchmark native TensorRT FP16 engine (ADOPTED)
python optimize/build_trt_engine.py --onnx results/onnx/wilor-fast.onnx \
  --out results/trt/wilor-fast-fp16.plan
python optimize/bench_trt.py --engine results/trt/wilor-fast-fp16.plan --iters 200

# 5. INT8 PTQ (accuracy-gated — REJECTED here)
python optimize/build_trt_int8.py --calib-count 64
python optimize/bench_trt_gate.py --fp16 results/trt/wilor-fast-fp16.plan \
  --int8 results/trt/wilor-fast-int8.plan --n 100
```

### Tests
```bash
python -m pytest tests/ -q                          # unit + property tests
python tests/audit_metrics_blackbox.py              # black-box known-answer audits
python tests/audit_aggregate_blackbox.py
python tests/audit_optimize_io_blackbox.py
python tests/audit_loader_blackbox.py                # needs FREIHAND_ROOT
```

---

## Repository layout
```
agents/          # the 00→12 runbook (brief → environment → models → benchmark → optimize → report)
benchmark/       # run_benchmark.py, aggregate.py (shared harness + decision matrix)
common/          # interface.py, metrics.py, profiling.py (BaseHandModel, PA-MPJPE, CUDA events)
data/            # FreiHandLoader, subset builder, synthetic egocentric clip generator
models/          # mediapipe / mobrecon / wilor / hamer wrappers (one per candidate)
optimize/        # export_onnx, validate_onnx, bench_onnx_latency, build_trt_engine,
                 # bench_trt, build_trt_int8, bench_trt_gate, bench_breakdown
tests/           # pytest unit + black-box/white-box audit executors
results/         # FINAL_REPORT.md, comparison_table.md, decision.md, raw/*.json, plots/
scripts/         # gpu_smoke_test.py
LICENSE          # MIT (code only — see DATA/WEIGHTS NOTICE below)
requirements.txt
pyproject.toml
```

---

## Honest limitations

- **Single dominant hand** protocol; no two-hand interaction benchmark.
- **MobRecon** accuracy is **provisional**: its checkpoint expects OpenMesh's half-edge-ordered spiral traversal; without a buildable OpenMesh wheel (conda-forge tops at py3.8, no compiler here) the exact winding can't be reproduced, so its 24.77 mm is not publishable. Latency/VRAM are real.
- **HaMeR** is **environment-blocked**: needs detectron2 compiled from source (no `nvcc` / no `sm_120` wheel). Demo data is fully extracted; only the detector CUDA ops can't build.
- **INT8 PTQ was rejected** by the accuracy gate (+3.89 mm vs FP16) — the MANO LBS parent-joint chain amplifies quantization error. FP16 is the deployed precision.
- **ORT TensorRT EP** is blocked by a version mismatch (ORT links `libnvinfer.so.10`, env has TRT 10.16 `cu12` which works for native engines but ORT 1.23.2's TRT EP expects the TRT 10 `.so.10` from the system path — use native TRT instead).

## Data / weights notice

This repository's **code** is MIT-licensed. The **assets it downloads** are not:
- **FreiHAND**: research/non-commercial use.
- **WiLoR weights**: CC-BY-NC-ND.
- **MANO**: its own license (manual registration required).
- **MediaPipe**: Apache-2.0 (the `.task` asset is redistributable under its terms).

None of these assets are committed to this repo (see `.gitignore`). By running the pipeline you accept their respective licenses.
