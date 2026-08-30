"""Final smoke test for the adopted WiLoR-fast PyTorch artifact.

Run in the WiLoR env with WILOR_ROOT set. Because native Windows MediaPipe and WSL GPU
models use different environments, this smoke test validates the deployed GPU artifact;
MediaPipe has its own validated native-Windows benchmark JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clip", default=str(ROOT / "data/egocentric_clips/clip_01_synthetic.mp4"))
    p.add_argument("--max-frames", type=int, default=150)
    args = p.parse_args()

    import torch
    from models.wilor_wrapper import WiLoRHandModel
    from data.freihand.loader import FreiHandLoader

    model = WiLoRHandModel(variant="fast")
    model.load("cuda")
    model.warmup(20)
    torch.cuda.reset_peak_memory_stats()

    # 5 held-out-from-calibration eval images (dev root contains the full 200-image eval subset).
    loader = FreiHandLoader(subset=ROOT / "data/freihand/subsets/dev.json")
    sample_ids = [0, 1, 2, 3, 4]
    for i in sample_ids:
        image, _, _ = loader[i]
        pred = model.infer(model.preprocess(image))
        assert pred and pred[0].joints_3d.shape == (21, 3)
        print(f"FreiHAND smoke {i}: PASS joints={pred[0].joints_3d.shape}")

    cap = cv2.VideoCapture(args.clip)
    if not cap.isOpened():
        raise FileNotFoundError(args.clip)
    times, mem = [], []
    misses = 0
    frame = 0
    while frame < args.max_frames:
        ok, image = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        pred = model.infer(model.preprocess(image))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
        mem.append(torch.cuda.memory_allocated() / 1e6)
        if pred:
            assert pred[0].joints_3d.shape == (21, 3)
        else:
            # Detector misses are valid on synthetic occluded frames; no-crash and
            # bounded memory are the smoke-test properties for the full pipeline.
            misses += 1
        frame += 1
    cap.release()
    if not times:
        raise RuntimeError("clip produced no frames")
    peak = torch.cuda.max_memory_allocated() / 1e6
    mean_ms = float(np.mean(times))
    p95_ms = float(np.percentile(times, 95))
    plateau_delta = float(max(mem) - min(mem))
    print(json.dumps({
        "clip": args.clip, "frames": frame, "detections": frame - misses, "misses": misses,
        "clip_detection_rate": (frame - misses) / frame if frame else 0.0,
        "mean_latency_ms": mean_ms,
        "p95_latency_ms": p95_ms, "torch_peak_allocated_mb": peak,
        "memory_range_mb": plateau_delta, "vram_ceiling_mb": 6000,
        "latency_target_ms": 33.333,
        "vram_pass": peak <= 6000,
        "latency_pass": mean_ms < 33.333,
        "no_growth_pass": plateau_delta < 512.0,
    }, indent=2))
    assert peak <= 6000, f"VRAM exceeded 6GB: {peak:.1f} MB"
    assert plateau_delta < 512.0, f"possible memory growth: {plateau_delta:.1f} MB"
    print("smoke test: PASS")


if __name__ == "__main__":
    main()
