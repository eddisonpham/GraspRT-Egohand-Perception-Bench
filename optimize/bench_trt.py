"""Benchmark the native TensorRT engine vs native PyTorch (WiLoR-fast).

Loads the .plan engine, runs the same FreiHAND crop, compares outputs against
the PyTorch reference (accuracy gate), and measures latency with CUDA events.

Usage (WSL, egohand env):
  LD_LIBRARY_PATH=<tensorrt_libs>:<nvidia libs> \
  WILOR_ROOT=~/src/WiLoR FREIHAND_ROOT=~/egohand_data/freihand \
    python optimize/bench_trt.py --engine results/trt/wilor-fast-fp16.plan
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optimize.bench_onnx_latency import summarize  # noqa: E402
from optimize.validate_onnx import load_crop  # noqa: E402


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=str(ROOT / "results" / "trt" / "wilor-fast-fp16.plan"))
    ap.add_argument("--variant", choices=["default", "fast"], default="fast")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--freihand-root", default=os.environ.get("FREIHAND_ROOT", ""))
    args = ap.parse_args()
    if not args.freihand_root:
        sys.exit("FREIHAND_ROOT not set and --freihand-root not given")

    import torch
    import tensorrt as trt

    from models.wilor_wrapper import WiLoRHandModel

    # PyTorch reference + crop.
    model = WiLoRHandModel(variant=args.variant)
    model.load("cuda")
    inner = model.model
    eager = getattr(inner.backbone, "_orig_mod", None)
    if eager is not None:
        inner.backbone = eager
    inner.requires_grad_(False)
    img_t, right_t = load_crop(model, args.freihand_root, args.index)
    dev = next(inner.parameters()).device
    right_tensor = torch.as_tensor(np.asarray(right_t, dtype=np.float32)).reshape(1)
    with torch.no_grad():
        out = inner.forward_step(
            {"img": img_t.to(dev, dtype=img_t.dtype), "right": right_tensor.to(dev)}, train=False
        )
    ref_j = out["pred_keypoints_3d"].float().cpu().numpy()
    ref_v = out["pred_vertices"].float().cpu().numpy()

    # TRT engine.
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    plan = Path(args.engine).read_bytes()
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        sys.exit("engine deserialize failed")
    ctx = engine.create_execution_context()
    print(f"engine IO: " + ", ".join(engine.get_tensor_name(i) for i in range(engine.num_io_tensors)))

    img_np = np.ascontiguousarray(img_t.cpu().numpy())
    img_dev = torch.from_numpy(img_np).to(dev)
    out_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                 if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
    out_bufs = {}
    for n in out_names:
        shape = tuple(ctx.get_tensor_shape(n))
        out_bufs[n] = torch.empty(shape, dtype=torch.float16, device=dev)
        ctx.set_tensor_address(n, int(out_bufs[n].data_ptr()))
    ctx.set_tensor_address(engine.get_tensor_name(0), int(img_dev.data_ptr()))

    def run():
        ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)

    # Warmup.
    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()

    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(args.iters):
        ev0.record()
        run()
        ev1.record()
        torch.cuda.synchronize()
        times.append(ev0.elapsed_time(ev1))
    s = summarize(times)
    print(f"trt:fp16: {s}")

    # Accuracy gate vs PyTorch reference.
    oj = out_bufs[out_names[0]].float().cpu().numpy()
    ov = out_bufs[out_names[1]].float().cpu().numpy()
    from common.metrics import procrustes_align
    dj = np.abs(oj - ref_j).max()
    dv = np.abs(ov - ref_v).max()
    aligned = procrustes_align(oj[0].astype(np.float64), ref_j[0].astype(np.float64))
    mpjpe = float(np.linalg.norm(aligned - ref_j[0], axis=-1).mean()) * 1000
    print(f"accuracy gate: joints max|d|={dj:.3e} verts max|d|={dv:.3e} PA-MPJPE={mpjpe:.3f}mm")

    raw = ROOT / "results" / "raw" / "trt-latency.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "engine": args.engine, "trt": trt.__version__,
        "latency": s,
        "gate": {"joints_max_abs_diff": float(dj), "verts_max_abs_diff": float(dv),
                 "pa_mpjpe_vs_pytorch_mm": mpjpe},
    }, indent=2))
    print(f"saved {raw}")
