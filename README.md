# GraspRT-Egohand-Perception-Bench

A measured benchmark + TensorRT optimization pipeline for hand pose/mesh models targeting a 6 GB VRAM GPU. Each candidate is scored on FreiHAND under a fixed decision matrix; the winner is exported to ONNX and quantized to a native TensorRT FP16 engine. Every reported number is measured on real data.

## Quick start

```bash
pip install -r requirements.txt                # torch/cu128 + project deps
export WILOR_ROOT=~/src/WiLoR
export FREIHAND_ROOT=~/egohand_data/freihand
python -m pytest tests/ -q                      # 81 pass / 5 skip (WSL-only loaders)
python benchmark/run_eval_suite.py              # full pipeline validation → results/eval-report.md
```

See [Setup](#setup) for full environment + asset requirements.

---

## Headline results

### Model comparison — FreiHAND dev (200 images)

| Model | Mean ms | FPS | PA-MPJPE mm | VRAM MB | Score |
|---|---:|---:|---:|---:|---:|
| **WiLoR-fast** 🏆 | **33.51** | **29.84** | **5.901** | **3,014** | **0.6942** |
| MediaPipe (CPU) | 34.35 | 29.11 | 15.883 | — | 0.6364 |
| MobRecon | 22.71 | 44.03 | 24.77* | 1,292 | 0.6000 |
| WiLoR default | 69.53 | 14.38 | 5.742 | 4,235 | 0.4000 |

### Optimization of the winner — reconstruction graph

| Backend | Mean ms | FPS | Accuracy |
|---|---:|---:|---|
| **TensorRT FP16** ✅ | **13.72** | **72.90** | 5.544 mm vs GT |
| Native PyTorch (FP16) | 24.01 | 41.65 | 5.9 mm (baseline) |
| ORT CUDA EP | 52.11 | 19.19 | 0.66 mm vs PyTorch |
| TRT INT8 PTQ | 10.30 | 97.05 | 9.438 mm — **rejected** |

**Decision:** WiLoR-fast + TRT FP16. 1.75× faster than native, inside VRAM budget, accuracy gate passed. INT8 honestly rejected (+3.89 mm — MANO LBS amplifies quantization error).

### Full pipeline vs optimization components

| Scope | Mean ms | FPS |
|---|---:|---:|
| TRT reconstruction alone | 12.6 | 79 |
| **End-to-end** (detector+crop+TRT) | **34.9** | **28.6** |
| Detector + crop share | 21.9 | — |

The reconstruction graph is 79 FPS, but the YOLO detector (42%) + ViTDet crop (20%) dominate end-to-end latency. That pre-optimization path is the current bottleneck, not the TRT engine.

---

## Architecture

```
FreiHandLoader → BaseHandModel.preprocess → BaseHandModel.infer → HandPrediction
                                                            ↓
                                    metrics (PA-MPJPE, PA-MPVPE, F-score)
                                                            ↓
                              results/raw/*.json → aggregate.py → decision + Pareto
                                                            ↓
                              WiLoR-fast → export_onnx → TRT FP16 engine → gate
```

| Module | Responsibility |
|---|---|
| `common/` | `interface` (HandPrediction + BaseHandModel), `metrics` (pure-NumPy alignment), `profiling` (CUDA-event latency, ResourceMonitor) |
| `data/` | FreiHandLoader, subset builder, synthetic egocentric clips |
| `models/` | one wrapper per candidate (mediapipe / mobrecon / wilor / hamer) |
| `benchmark/` | `run_benchmark` (one model/process), `aggregate` (weighted scoring), `run_eval_suite` (validation) |
| `optimize/` | ONNX export + validate, TRT build + bench + gate, INT8, batch/detector experiments |
| `scripts/` | `gpu_smoke_test`, `smoke_test_egocentric` |
| `tests/` | 81 unit + black-box + white-box tests |

Full detail: [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Setup

### 1. System

- NVIDIA GPU, CUDA 12.x, Blackwell/Ada/Ampere, **6+ GB VRAM**.
- Windows + WSL2 for GPU/TensorRT; native Windows only for MediaPipe.
- Driver ≥ 575.x.

### 2. Conda environment (WSL)

```bash
conda create -n egohand python=3.10 -y && conda activate egohand
pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### 3. Assets (manual, license-restricted)

```bash
export WILOR_ROOT=$HOME/src/WiLoR            # git clone https://github.com/rolpotamias/WiLoR
export MOBRECON_ROOT=$HOME/src/HandMesh      # git clone https://github.com/SeanChenxy/HandMesh
export FREIHAND_ROOT=$HOME/egohand_data/freihand
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

Download FreiHAND, MANO (`MANO_RIGHT.pkl`, register at the MANO site), and WiLoR checkpoints per `data/freihand/` + each repo's README.

### 4. Known issues / pinning

| Issue | Fix |
|---|---|
| `chumpy` breaks on numpy≥2 | Apply the alias patch (below) |
| default `tensorrt` wheel is CUDA-13 (fails `cudaError 35`) | Pin `tensorrt-cu12==10.16.1.11` |
| HaMeR won't compile detectron2 | No `sm_120` wheel; needs `nvcc` (environment-blocked) |

```bash
# chumpy numpy2 patch
python -c "p=__import__('chumpy').__file__.replace('/__init__.py',''); s=open(p).read(); open(p,'w').write(s.replace('from numpy import bool, int, float, complex, object, unicode, str, nan, inf','from numpy import nan, inf'))"
```

---

## Usage

### Benchmark

```bash
python benchmark/run_benchmark.py --model wilor --variant fast --iters 200 --subset dev
python benchmark/run_benchmark.py --model mobrecon --iters 200 --subset dev   # WSL
python benchmark/run_benchmark.py --model mediapipe --iters 200 --subset dev  # Windows
python benchmark/aggregate.py           # → comparison_table.md, decision.md, pareto.png
python benchmark/run_eval_suite.py      # → results/eval-report.md (validates everything)
```

### Optimize the winner

```bash
python optimize/export_onnx.py --variant fast            # → results/onnx/wilor-fast.onnx
python optimize/validate_onnx.py                          # Δ vs PyTorch (CPU/CUDA EP)
python optimize/build_trt_engine.py && python optimize/bench_trt.py --iters 200
python optimize/bench_trt_gate.py --n 100                 # accuracy gate vs GT
python optimize/bench_trt_profiling.py --iters 200        # GPU util/power/temp
python optimize/bench_e2e_pipeline.py --n-images 100      # true wall-clock FPS
```

### Experiments (documented results)

```bash
python optimize/bench_detector.py         # detector FP16/imgsz — no gain (neutral)
python optimize/bench_batch_throughput.py # multi-stream — GPU saturated at batch=1
python optimize/bench_breakdown.py        # per-stage latency
python scripts/smoke_test_egocentric.py   # synthetic clips, behavior invariants
```

### Tests

```bash
python -m pytest tests/ -q                # 81 pass / 5 skip
python benchmark/run_eval_suite.py        # pytest + JSON schema + aggregate + artifacts
```

---

## Honest limitations

| Item | Status | Why |
|---|---|---|
| MobRecon accuracy | provisional | OpenMesh half-edge spiral order not reproducible without a compiler (24.77 vs published 6.9 mm) |
| HaMeR | environment-blocked | detectron2 needs `nvcc`; no `sm_120` wheel |
| INT8 PTQ | rejected | +3.89 mm vs FP16; MANO LBS amplifies error |
| End-to-end 30 FPS | not met | detector+crop (22 ms) dominates; TRT recon alone is 79 FPS |
| Multi-stream | no gain | GPU SMs already saturated at single stream (79% util) |
| Single hand | protocol | one dominant hand per frame; no two-hand interaction |

---

## License

**Code:** MIT. **Downloaded assets** (FreiHAND, WiLoR weights, MANO) carry stricter terms and are not committed.

Full measured details: [`results/FINAL_REPORT.md`](results/FINAL_REPORT.md).