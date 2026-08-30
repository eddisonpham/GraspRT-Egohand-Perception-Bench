"""Latency benchmark: WiLoR-fast ONNX graph on ORT (CPU/CUDA EP) vs native PyTorch.

Fixed-shape reconstruction graph only (detector/crop outside). Uses CUDA events
for GPU timing on CUDA paths and perf_counter for CPU. Reports mean/p50/p95/p99
over timed iterations after warmup. Results append to results/raw/ort-latency.json.

Usage (WSL, egohand env):
  WILOR_ROOT=~/src/WiLoR FREIHAND_ROOT=~/egohand_data/freihand \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 LD_LIBRARY_PATH=<nvidia libs> \
    python optimize/bench_onnx_latency.py --iters 200
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optimize.validate_onnx import load_crop  # noqa: E402


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
    return s[k]


def summarize(values_ms: list[float]) -> dict:
    return {
        "mean_ms": round(statistics.fmean(values_ms), 3),
        "p50_ms": round(pct(values_ms, 50), 3),
        "p95_ms": round(pct(values_ms, 95), 3),
        "p99_ms": round(pct(values_ms, 99), 3),
        "fps": round(1000 / statistics.fmean(values_ms), 2),
        "n": len(values_ms),
    }


def bench_ort(path: str, feed: dict, providers: list[str], warmup: int, iters: int) -> dict | None:
    import onnxruntime as ort

    try:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        sess = ort.InferenceSession(path, opts, providers=providers)
    except Exception as exc:
        print(f"ORT session ({providers[0]}) FAILED: {exc!r}")
        return None
    active = sess.get_providers()
    names = {i.name for i in sess.get_inputs()}
    run_feed = {k: v for k, v in feed.items() if k in names}

    for _ in range(warmup):
        sess.run(None, run_feed)
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, run_feed)
        times.append((time.perf_counter() - t0) * 1000)
    label = f"ort:{active[0]}"
    print(f"{label}: {summarize(times)}")
    return {"label": label, **summarize(times)}


def bench_native(inner, img_t, right_tensor, warmup: int, iters: int) -> dict:
    import torch

    dev = next(inner.parameters()).device
    img = img_t.to(dev, dtype=img_t.dtype)
    right = right_tensor.to(dev)

    def step():
        with torch.no_grad():
            inner.forward_step({"img": img, "right": right}, train=False)

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    times: list[float] = []
    for _ in range(iters):
        ev0.record()
        step()
        ev1.record()
        torch.cuda.synchronize()
        times.append(ev0.elapsed_time(ev1))
    print(f"native:pytorch: {summarize(times)}")
    return {"label": "native:pytorch", **summarize(times)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(ROOT / "results" / "onnx" / "wilor-fast.onnx"))
    ap.add_argument("--variant", choices=["default", "fast"], default="fast")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--freihand-root", default=os.environ.get("FREIHAND_ROOT", ""))
    vargs = ap.parse_args()
    if not vargs.freihand_root:
        sys.exit("FREIHAND_ROOT not set and --freihand-root not given")

    import torch

    from models.wilor_wrapper import WiLoRHandModel

    model = WiLoRHandModel(variant=vargs.variant)
    model.load("cuda")
    inner = model.model
    eager = getattr(inner.backbone, "_orig_mod", None)
    if eager is not None:
        inner.backbone = eager
    inner.requires_grad_(False)

    img_t, right_t = load_crop(model, vargs.freihand_root, vargs.index)
    right_tensor = torch.as_tensor(np.asarray(right_t, dtype=np.float32)).reshape(1)

    img_np = img_t.cpu().numpy()
    right_np = right_tensor.numpy()
    feed = {"image": img_np, "right": right_np}

    results = [bench_native(inner, img_t, right_tensor, vargs.warmup, vargs.iters)]
    for providers in (["CUDAExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"]):
        r = bench_ort(vargs.onnx, feed, providers, vargs.warmup, vargs.iters)
        if r:
            results.append(r)

    raw = ROOT / "results" / "raw" / "ort-latency.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(),
               "onnx": vargs.onnx, "variant": vargs.variant, "results": results}
    raw.write_text(json.dumps(payload, indent=2))
    print(f"saved {raw}")
