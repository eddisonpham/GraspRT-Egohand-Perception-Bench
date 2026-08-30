"""Detector optimization experiment: YOLO hand detector latency vs precision/size.

Measures the WiLoR YOLO hand detector alone across configurations:
  - baseline (FP32, native imgsz, conf=0.3) — the current wrapper settings
  - half=True (FP16)
  - imgsz 640/512/448/384 (detector input size)
  - combined best (FP16 + smaller imgsz)
Accuracy proxy: detection count + avg confidence on the same 30 FreiHAND
images, so a faster config isn't silently dropping hands.

Usage (WSL, egohand env):
  WILOR_ROOT=~/src/WiLoR FREIHAND_ROOT=~/egohand_data/freihand \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python optimize/bench_detector.py
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

WARMUP = 10
ITERS = 40
N_IMAGES = 30


def summarize(times_ms: list[float]) -> dict:
    arr = np.asarray(times_ms)
    return {
        "mean_ms": round(float(arr.mean()), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "n": len(times_ms),
    }


def main() -> None:
    import torch
    from ultralytics import YOLO

    from data.freihand.loader import FreiHandLoader
    from models.wilor_wrapper import WiLoRHandModel

    model = WiLoRHandModel(variant="fast")
    model.load("cuda")
    detector: YOLO = model.detector

    loader = FreiHandLoader(subset="data/freihand/subsets/dev.json")
    images = [loader[i][0] for i in range(N_IMAGES)]

    # Detection-count baseline (accuracy proxy): confidences at baseline settings.
    def detect_stats(**kwargs) -> tuple[float, float]:
        counts, confs = [], []
        for img in images:
            res = detector(img, verbose=False, **kwargs)[0]
            n = 0 if res.boxes is None else len(res.boxes)
            counts.append(1 if n > 0 else 0)
            if n:
                confs.append(float(res.boxes.conf.max()))
        return float(np.mean(counts)), float(np.mean(confs)) if confs else 0.0

    configs = [
        ("baseline-fp32", {}),
        ("fp16", {"half": True}),
        ("imgsz512", {"imgsz": 512}),
        ("imgsz448", {"imgsz": 448}),
        ("imgsz384", {"imgsz": 384}),
        ("fp16+imgsz512", {"half": True, "imgsz": 512}),
        ("fp16+imgsz448", {"half": True, "imgsz": 448}),
        ("fp16+imgsz384", {"half": True, "imgsz": 384}),
    ]

    results = []
    for name, kwargs in configs:
        for _ in range(WARMUP):
            detector(images[0], verbose=False, **kwargs)
        torch.cuda.synchronize()
        times = []
        for img in images:
            t0 = time.perf_counter()
            detector(img, verbose=False, **kwargs)
            times.append((time.perf_counter() - t0) * 1000)
        torch.cuda.synchronize()
        # accuracy proxy on the same config
        det_rate, avg_conf = detect_stats(**kwargs)
        s = summarize(times)
        results.append({"config": name, "detection_rate": det_rate,
                        "avg_max_conf": round(avg_conf, 4), **s})
        print(f"{name:16s} {s['mean_ms']:7.2f} ms  p95 {s['p95_ms']:7.2f}  "
              f"det {det_rate:.2f}  conf {avg_conf:.3f}")

    out = ROOT / "results" / "raw" / "detector-tuning.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_images": N_IMAGES, "iters": ITERS, "results": results,
    }, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
