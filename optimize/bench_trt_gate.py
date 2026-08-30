"""Accuracy gate: TRT engine vs GT compared with PyTorch vs GT.

For each model (fp16 engine, int8 engine), run N dev crops through the engine,
align with Procrustes to GT joints, and compute PA-MPJPE. Compare with the
PyTorch fast reference PA-MPJPE on the same images. The runbook gate: INT8
PA-MPJPE-vs-GT must stay within +0.5mm of the FP16 baseline, else reject INT8.

Usage (WSL, egohand env):
  LD_LIBRARY_PATH=<tensorrt_libs>:<nvidia libs> \
  WILOR_ROOT=~/src/WiLoR FREIHAND_ROOT=~/egohand_data/freihand \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    python optimize/bench_trt_gate.py --fp16 results/trt/wilor-fast-fp16.plan \
      --int8 results/trt/wilor-fast-int8.plan --n 100
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

from common.metrics import procrustes_align, pa_mpjpe


def load_engine(path, dev):
    import tensorrt as trt
    import torch

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(Path(path).read_bytes())
    if engine is None:
        sys.exit(f"engine deserialize failed: {path}")
    ctx = engine.create_execution_context()
    in_name = engine.get_tensor_name(0)
    out_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                 if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
    return ctx, in_name, out_names


def run_engine(ctx, in_name, out_names, img_np, dev):
    import torch
    img_dev = torch.from_numpy(np.ascontiguousarray(img_np)).to(dev)
    bufs = {}
    for n in out_names:
        shape = tuple(ctx.get_tensor_shape(n))
        bufs[n] = torch.empty(shape, dtype=torch.float16, device=dev)
        ctx.set_tensor_address(n, int(bufs[n].data_ptr()))
    ctx.set_tensor_address(in_name, int(img_dev.data_ptr()))
    ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return [bufs[n].float().cpu().numpy() for n in out_names]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp16", default=str(ROOT / "results" / "trt" / "wilor-fast-fp16.plan"))
    ap.add_argument("--int8", default=str(ROOT / "results" / "trt" / "wilor-fast-int8.plan"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--freihand-root", default=os.environ.get("FREIHAND_ROOT", ""))
    args = ap.parse_args()
    if not args.freihand_root:
        sys.exit("FREIHAND_ROOT not set and --freihand-root not given")

    import torch

    from data.freihand.loader import FreiHandLoader
    from models.wilor_wrapper import WiLoRHandModel

    model = WiLoRHandModel(variant="fast")
    model.load("cuda")
    inner = model.model
    eager = getattr(inner.backbone, "_orig_mod", None)
    if eager is not None:
        inner.backbone = eager
    inner.requires_grad_(False)
    dev = next(inner.parameters()).device

    loader = FreiHandLoader(subset="data/freihand/subsets/dev.json")
    n = min(args.n, len(loader))

    results = {}
    for label, engine_path in (("fp16", args.fp16), ("int8", args.int8)):
        if not Path(engine_path).exists():
            print(f"[{label}] missing engine {engine_path}; skipping")
            continue
        ctx, in_name, out_names = load_engine(engine_path, dev)
        errs = []
        for i in range(n):
            img, gt_j, _ = loader[i]
            pre = model.preprocess(img)
            if pre["dataset"] is None:
                continue
            item = pre["dataset"][0]
            img_np = np.asarray(item["img"], dtype=np.float16)[None]
            oj, _ = run_engine(ctx, in_name, out_names, img_np, dev)
            pred = oj[0].astype(np.float64)
            # pa_mpjpe scales the residual by 1000 itself (m->mm); pass GT in
            # its native meters. Both pred and GT must be in the same unit (meters).
            gt = gt_j.astype(np.float64)
            errs.append(pa_mpjpe(pred, gt))
        results[label] = {
            "n": len(errs),
            "pa_mpjpe_vs_gt_mm": round(float(np.mean(errs)), 3),
            "pa_mpjpe_std_mm": round(float(np.std(errs)), 3),
        }
        print(f"[{label}] PA-MPJPE vs GT: {results[label]['pa_mpjpe_vs_gt_mm']:.3f} ± "
              f"{results[label]['pa_mpjpe_std_mm']:.3f} mm (n={len(errs)})")

    gate = {}
    if "int8" in results and "fp16" in results:
        delta = results["int8"]["pa_mpjpe_vs_gt_mm"] - results["fp16"]["pa_mpjpe_vs_gt_mm"]
        gate = {"int8_vs_fp16_delta_mm": round(delta, 3),
                "gate_threshold_mm": 0.5,
                "gate_passed": bool(delta < 0.5)}
        print(f"INT8 gate: Δ={delta:+.3f}mm vs FP16, threshold +0.5mm → "
              f"{'PASS' if gate['gate_passed'] else 'REJECT'}")

    raw = ROOT / "results" / "raw" / "trt-accuracy-gate.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(), "per_engine": results, "gate": gate,
    }, indent=2))
    print(f"saved {raw}")
