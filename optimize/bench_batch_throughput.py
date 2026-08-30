"""Batch throughput benchmark via concurrent CUDA streams.

Instead of rebuilding the ONNX/TRT graph with dynamic batch (which breaks
TRT parsing due to coupled reshape ops), we run the existing batch=1 engine
on N concurrent CUDA streams. This fills the GPU SMs the same way batching
does and measures real concurrent throughput.

Measures:
  - Single-stream baseline (batch=1, sequential)
  - 2/4/8 concurrent streams (each running batch=1)
  - GPU utilization during concurrent runs (via ResourceMonitor)

Usage (WSL, egohand env):
  WILOR_ROOT=~/src/WiLoR FREIHAND_ROOT=~/egohand_data/freihand \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python optimize/bench_batch_throughput.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STREAM_COUNTS = [1, 2, 4, 8]
WARMUP = 20
ITERS = 200


def summarize(times_ms: list[float]) -> dict:
    arr = np.asarray(times_ms)
    return {
        "mean_ms": round(float(arr.mean()), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "std_ms": round(float(arr.std()), 3),
    }


def load_engine(engine_path: str):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


def benchmark_single_stream(engine, img_np: np.ndarray) -> dict:
    """Standard single-stream sequential benchmark."""
    import torch

    context = engine.create_execution_context()
    d_input = torch.from_numpy(img_np).cuda()
    d_joints = torch.empty((1, 21, 3), dtype=torch.float16, device="cuda")
    d_verts = torch.empty((1, 778, 3), dtype=torch.float16, device="cuda")

    def infer():
        context.execute_v2(bindings=[
            d_input.data_ptr(), d_joints.data_ptr(), d_verts.data_ptr()])
        torch.cuda.synchronize()

    for _ in range(WARMUP):
        infer()

    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        infer()
        times.append((time.perf_counter() - t0) * 1000)

    s = summarize(times)
    return {
        "mode": "single-stream",
        "n_streams": 1,
        "batch_per_stream": 1,
        "total_samples_per_iter": 1,
        "total_ms": s["mean_ms"],
        "per_sample_ms": s["mean_ms"],
        "throughput_fps": round(1000 / s["mean_ms"], 1),
        "p95_ms": s["p95_ms"],
        "iters": ITERS,
    }


def benchmark_multi_stream(engine, n_streams: int, img_np: np.ndarray) -> dict:
    """Run N independent CUDA streams, each with its own context + buffers."""
    import torch

    # Create N independent execution contexts + buffer sets
    streams = []
    contexts = []
    for _ in range(n_streams):
        s = torch.cuda.Stream()
        streams.append(s)
        ctx = engine.create_execution_context()
        contexts.append(ctx)

    # Allocate per-stream buffers
    buf_pairs = []
    for _ in range(n_streams):
        d_in = torch.from_numpy(img_np).cuda()
        d_j = torch.empty((1, 21, 3), dtype=torch.float16, device="cuda")
        d_v = torch.empty((1, 778, 3), dtype=torch.float16, device="cuda")
        buf_pairs.append((d_in, d_j, d_v))

    def infer_all():
        for i in range(n_streams):
            with torch.cuda.stream(streams[i]):
                d_in, d_j, d_v = buf_pairs[i]
                contexts[i].execute_v2(bindings=[
                    d_in.data_ptr(), d_j.data_ptr(), d_v.data_ptr()])
        torch.cuda.synchronize()

    for _ in range(WARMUP):
        infer_all()

    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        infer_all()
        times.append((time.perf_counter() - t0) * 1000)

    s = summarize(times)
    total_samples = n_streams
    per_sample_ms = s["mean_ms"] / total_samples
    throughput_fps = 1000.0 / per_sample_ms
    return {
        "mode": f"{n_streams}-stream",
        "n_streams": n_streams,
        "batch_per_stream": 1,
        "total_samples_per_iter": total_samples,
        "total_ms": s["mean_ms"],
        "per_sample_ms": round(per_sample_ms, 3),
        "throughput_fps": round(throughput_fps, 1),
        "p95_ms": s["p95_ms"],
        "iters": ITERS,
    }


def main() -> None:
    from common.profiling import ResourceMonitor
    from models.wilor_wrapper import WiLoRHandModel
    from optimize.validate_onnx import load_crop

    engine_path = str(ROOT / "results" / "trt" / "wilor-fast-fp16.plan")
    if not os.path.exists(engine_path):
        sys.exit(f"Engine not found at {engine_path}")

    model = WiLoRHandModel(variant="fast")
    model.load("cuda")

    img_t, _ = load_crop(model, str(os.environ.get("FREIHAND_ROOT", "")), 0)
    img_np = img_t.cpu().numpy().astype(np.float16)

    import torch
    engine = load_engine(engine_path)

    results = []
    for n in STREAM_COUNTS:
        print(f"\n--- {n} stream(s) ---")
        mon = ResourceMonitor(interval_s=0.05)
        mon.start()
        torch.cuda.reset_peak_memory_stats()

        if n == 1:
            r = benchmark_single_stream(engine, img_np)
        else:
            r = benchmark_multi_stream(engine, n, img_np)

        mon.stop()
        vram_peak = round(torch.cuda.max_memory_allocated() / 1e6, 1)
        r["resource"] = mon.summary()
        r["vram_peak_mb"] = vram_peak
        results.append(r)
        print(f"  {r['total_ms']:.1f} ms total, "
              f"{r['per_sample_ms']:.1f} ms/sample, "
              f"{r['throughput_fps']:.0f} FPS throughput, "
              f"vram {vram_peak:.0f} MB")
        if "gpu_util_pct_mean" in r["resource"]:
            print(f"  GPU util: {r['resource']['gpu_util_pct_mean']:.0f}% "
                  f"(peak {r['resource']['gpu_util_pct_peak']:.0f}%), "
                  f"power: {r['resource'].get('power_watts_mean', 0):.0f}W")

    # Summary table
    print("\n" + "=" * 85)
    print(f"{'Streams':>7} {'Total ms':>10} {'Per-sample ms':>15} "
          f"{'FPS throughput':>15} {'VRAM MB':>8} {'GPU util':>9}")
    print("-" * 85)
    for r in results:
        gpu = r["resource"].get("gpu_util_pct_mean", 0)
        print(f"{r['n_streams']:>7} {r['total_ms']:>10.1f} "
              f"{r['per_sample_ms']:>15.1f} {r['throughput_fps']:>15.0f} "
              f"{r['vram_peak_mb']:>8.0f} {gpu:>8.0f}%")
    print("=" * 85)

    # Speedup over single-stream
    if results[0]["throughput_fps"] > 0:
        base = results[0]["throughput_fps"]
        print(f"\nSpeedup over single-stream:")
        for r in results:
            speedup = r["throughput_fps"] / base
            print(f"  {r['n_streams']} stream(s): {speedup:.2f}x")

    out = ROOT / "results" / "raw" / "batch-throughput.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "engine": engine_path,
        "warmup": WARMUP, "iters": ITERS,
        "results": results,
    }, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
