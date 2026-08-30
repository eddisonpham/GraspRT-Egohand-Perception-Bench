# GraspRT-Egohand-Perception-Bench

Controlled benchmark and TensorRT optimization pipeline for hand pose / mesh models on a constrained GPU (6 GB VRAM). Four candidates are measured against FreiHAND under a fixed decision matrix; the winner is exported to ONNX and quantized to a TensorRT FP16 engine. Every reported number is measured.

## Results

**Model comparison** (FreiHAND dev, 200 images)

| Model | Mean ms | FPS | PA-MPJPE mm | VRAM MB | Score |
|---|---:|---:|---:|---:|---:|
| **WiLoR-fast** | **33.51** | **29.84** | **5.901** | **3,014** | **0.6942** |
| MediaPipe (CPU) | 34.35 | 29.11 | 15.883 | — | 0.6364 |
| MobRecon | 22.71 | 44.03 | 24.77* | 1,292 | 0.6000 |
| WiLoR default | 69.53 | 14.38 | 5.742 | 4,235 | 0.4000 |

**Optimization of the winner** (reconstruction graph)

| Backend | Mean ms | FPS | Accuracy |
|---|---:|---:|---|
| **TensorRT FP16** | **13.72** | **72.90** | 5.544 mm vs GT ✅ |
| Native PyTorch (FP16) | 24.01 | 41.65 | 5.9 mm (baseline) |
| ORT CUDA EP | 52.11 | 19.19 | 0.66 mm vs PyTorch |
| TRT INT8 PTQ | 10.30 | 97.05 | 9.438 mm — **REJECTED** (+3.89 mm) |

TRT FP16 is the deployed engine: 1.75× faster than native, inside the VRAM budget, gate passed. INT8 was honestly rejected.

Full report: [`results/FINAL_REPORT.md`](results/FINAL_REPORT.md). Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Setup

**System:** NVIDIA GPU (CUDA 12.x, Blackwell/Ada/Ampere), 6+ GB VRAM. Windows with WSL2 for GPU work; native Windows only for MediaPipe.

```bash
conda create -n egohand python=3.10 -y && conda activate egohand
pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Set environment variables and download license-restricted assets (WiLoR, MobRecon, FreiHAND, MANO) per `agents/04_DATASET_PREP.md`:

```bash
export WILOR_ROOT=$HOME/src/WiLoR
export MOBRECON_ROOT=$HOME/src/HandMesh
export FREIHAND_ROOT=$HOME/egohand_data/freihand
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

**Known issues:** `chumpy` needs a numpy≥2 patch; `tensorrt-cu12==10.16.1.11` must be pinned (default wheel is a CUDA-13 build); HaMeR needs detectron2 compiled from source (no `nvcc` here).

## Usage

```bash
# Benchmark
python benchmark/run_benchmark.py --model wilor --variant fast --iters 200 --subset dev
python benchmark/aggregate.py

# Optimize the winner
python optimize/export_onnx.py --variant fast
python optimize/validate_onnx.py
python optimize/build_trt_engine.py && python optimize/bench_trt.py --iters 200

# Tests
python -m pytest tests/ -q
```

## Limitations

- **MobRecon** accuracy is provisional (24.77 mm*): its checkpoint expects OpenMesh's half-edge spiral ordering, not reproducible without a compiler. Latency/VRAM are real.
- **HaMeR** is environment-blocked: detectron2 won't compile without `nvcc` / an `sm_120` wheel.
- **INT8 PTQ rejected**: the MANO LBS chain amplifies quantization error (+3.89 mm).
- Single dominant hand protocol; synthetic egocentric clips are qualitative-only.

## License

MIT for the code. Downloaded assets (FreiHAND, WiLoR weights, MANO) carry their own stricter terms and are not committed.
