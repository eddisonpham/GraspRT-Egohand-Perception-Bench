"""Latency breakdown for the WiLoR-fast end-to-end pipeline.

Measures each stage separately on the same images with CUDA events (GPU stages)
and perf_counter (CPU stages): preprocessing+detector, crop/collate, model
forward, and postprocess. Writes results/raw/latency-breakdown.json.

Usage (WSL, egohand env):
  WILOR_ROOT=~/src/WiLoR FREIHAND_ROOT=~/egohand_data/freihand \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python optimize/bench_breakdown.py
"""
from __future__ import annotations

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


def pct(values, p):
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
    return s[k]


def summ(values):
    return {
        "mean_ms": round(statistics.fmean(values), 3),
        "p95_ms": round(pct(values, 95), 3),
        "n": len(values),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-images", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

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

    loader = FreiHandLoader(subset="data/freihand/subsets/dev.json")
    n_images = min(args.n_images, 50)
    images = [loader[i][0] for i in range(n_images)]

    WARMUP, ITERS = args.warmup, args.iters

    # --- Stage 1: preprocess + detector (CPU + GPU detector call) ---
    times_det = []
    for i in range(ITERS):
        img = images[i % n_images]
        t0 = time.perf_counter()
        pre = model.preprocess(img)
        times_det.append((time.perf_counter() - t0) * 1000)
    del times_det[:WARMUP]

    # --- Stage 2: crop/collate (dataset __getitem__) ---
    ds = pre["dataset"]
    times_crop = []
    for i in range(ITERS + WARMUP):
        t0 = time.perf_counter()
        item = ds[i % len(ds)]
        times_crop.append((time.perf_counter() - t0) * 1000)
    del times_crop[:WARMUP]

    # --- Stage 3: model forward (CUDA events) ---
    item = ds[0]
    item = {k: torch.as_tensor(v) if isinstance(v, np.ndarray) else v for k, v in item.items()}
    item = {
        k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.ndim >= 1 and k in {"img", "right", "box_center", "box_size", "img_size"} else v)
        for k, v in item.items()
    }
    from wilor.utils import recursive_to

    item = recursive_to(item, model._torch_device)
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    times_fwd = []
    for i in range(ITERS + WARMUP):
        torch.cuda.synchronize()
        ev0.record()
        with torch.no_grad():
            inner.forward_step(item, train=False)
        ev1.record()
        torch.cuda.synchronize()
        times_fwd.append(ev0.elapsed_time(ev1))
    del times_fwd[:WARMUP]

    # --- Stage 4: postprocess (CPU tensor->numpy) ---
    with torch.no_grad():
        out = inner.forward_step(item, train=False)
    times_post = []
    for _ in range(ITERS + WARMUP):
        t0 = time.perf_counter()
        _ = out["pred_keypoints_3d"][0].detach().float().cpu().numpy()
        _ = out["pred_vertices"][0].detach().float().cpu().numpy()
        times_post.append((time.perf_counter() - t0) * 1000)
    del times_post[:WARMUP]

    breakdown = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": "wilor-fast",
        "preprocess_plus_detector": summ(times_det),
        "crop_collate": summ(times_crop),
        "model_forward": summ(times_fwd),
        "postprocess": summ(times_post),
    }
    total = sum(breakdown[k]["mean_ms"] for k in
                ("preprocess_plus_detector", "crop_collate", "model_forward", "postprocess"))
    breakdown["sum_of_stages_ms"] = round(total, 3)
    print(json.dumps(breakdown, indent=2))

    raw = ROOT / "results" / "raw" / "latency-breakdown.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps(breakdown, indent=2))
    print(f"saved {raw}")
