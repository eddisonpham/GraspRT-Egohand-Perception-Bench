"""TRT FP16 engine benchmark with full production profiling.

Measures latency + GPU util + power draw + CPU load on real FreiHAND images.
Usage (WSL, egohand env):
  WILOR_ROOT=~/src/WiLoR FREIHAND_ROOT=~/egohand_data/freihand \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python optimize/bench_trt_profiling.py \
    --engine results/trt/wilor-fast-fp16.plan --iters 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    import tensorrt as trt

    from common.profiling import ResourceMonitor, write_raw
    from data.freihand.loader import FreiHandLoader
    from models.wilor_wrapper import WiLoRHandModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=str(ROOT / "results" / "trt" / "wilor-fast-fp16.plan"))
    ap.add_argument("--variant", default="fast")
    ap.add_argument("--n-images", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    # --- Load TRT engine ---
    import torch
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine_path = args.engine
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # --- Prepare input ---
    model = WiLoRHandModel(variant=args.variant)
    model.load("cuda")
    eager = getattr(model.model.backbone, "_orig_mod", None)
    if eager is not None:
        model.model.backbone = eager
    model.model.requires_grad_(False)

    from optimize.validate_onnx import load_crop
    img_t, _ = load_crop(model, str(os.environ.get("FREIHAND_ROOT", "")), 0)
    img_np = img_t.cpu().numpy().astype(np.float16)

    # Allocate torch CUDA buffers and map via data_ptr for TRT
    d_input = torch.from_numpy(np.ascontiguousarray(img_np)).cuda()
    d_joints = torch.empty((1, 21, 3), dtype=torch.float32, device="cuda")
    d_verts = torch.empty((1, 778, 3), dtype=torch.float32, device="cuda")
    stream = torch.cuda.Stream()

    def trt_infer():
        context.execute_v2(
            bindings=[d_input.data_ptr(), d_joints.data_ptr(), d_verts.data_ptr()])
        stream.synchronize()

    # --- Warmup ---
    for _ in range(args.warmup):
        trt_infer()

    # --- Production profiling ---
    mon = ResourceMonitor(interval_s=0.05)
    mon.start()
    import torch
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        trt_infer()
        times.append((time.perf_counter() - t0) * 1000)
    mon.stop()

    arr = np.asarray(times)
    stats = {
        "mean_ms": round(float(arr.mean()), 3),
        "median_ms": round(float(np.median(arr)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "std_ms": round(float(arr.std()), 3),
        "fps": round(1000 / float(arr.mean()), 2),
        "n": len(times),
    }
    resource = mon.summary()
    vram = {
        "torch_peak_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
    }

    print(f"TRT FP16 engine: {stats}")
    print(f"Resource profile: {resource}")

    payload = {
        "model": "wilor-fast",
        "variant": "fast",
        "backend": "tensorrt-fp16",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "engine": engine_path,
        "latency_ms": stats,
        "resource": resource,
        "vram_mb": vram,
        "protocol": {"warmup": args.warmup, "iters": args.iters, "n_images": args.n_images},
    }
    out = ROOT / "results" / "raw" / "trt-fp16-profiled.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
