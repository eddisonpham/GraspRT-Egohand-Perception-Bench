"""Test torch.compile on the YOLO detector: latency + box equivalence.

torch.compile accelerates the conv backbone compute while keeping the exact
native Pose-decode + NMS semantics (they stay outside the compiled unit), so
boxes cannot drift the way the TRT full-model export did. This is Lead B.
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


def main() -> None:
    import torch

    from data.freihand.loader import FreiHandLoader
    from models.wilor_wrapper import WiLoRHandModel

    model = WiLoRHandModel(variant="fast")
    model.load("cuda")
    det = model.detector
    detector_model = det.model  # PoseModel

    loader = FreiHandLoader(subset="data/freihand/subsets/dev.json")
    images = [loader[i][0] for i in range(30)]

    # --- Baseline eager boxes + latency ---
    for img in images[:5]:
        det(img, verbose=False)
    eager_boxes, eager_times = [], []
    for img in images:
        t0 = time.perf_counter()
        r = det(img, verbose=False)[0]
        eager_times.append((time.perf_counter() - t0) * 1000)
        eager_boxes.append(r.boxes.xyxy.detach().cpu().numpy() if r.boxes is not None else None)

    # --- Compile and re-run (same predict path) ---
    print("Compiling detector model with torch.compile ...")
    compiled = torch.compile(detector_model)
    det.model = compiled
    for img in images[:5]:
        det(img, verbose=False)
    torch.cuda.synchronize()
    compiled_boxes, compiled_times = [], []
    for img in images:
        t0 = time.perf_counter()
        r = det(img, verbose=False)[0]
        compiled_times.append((time.perf_counter() - t0) * 1000)
        compiled_boxes.append(r.boxes.xyxy.detach().cpu().numpy() if r.boxes is not None else None)

    # --- Box equivalence ---
    diffs = []
    for eb, cb in zip(eager_boxes, compiled_boxes):
        if eb is None or cb is None:
            continue
        n = min(len(eb), len(cb))
        diffs.append(float(np.abs(eb[:n] - cb[:n]).max()))

    def summ(arr):
        a = np.asarray(arr)
        return {"mean_ms": round(float(a.mean()), 3),
                "p95_ms": round(float(np.percentile(a, 95)), 3)}

    e = summ(eager_times); c = summ(compiled_times)
    print("\n=== torch.compile YOLO detector ===")
    print(f"  eager    mean {e['mean_ms']:7.3f}  p95 {e['p95_ms']:7.3f}")
    print(f"  compiled mean {c['mean_ms']:7.3f}  p95 {c['p95_ms']:7.3f}")
    print(f"  speedup {e['mean_ms']/c['mean_ms']:.2f}x")
    print(f"  mean max-abs box diff: {np.mean(diffs):.4f} px  (n={len(diffs)})")

    out = ROOT / "results" / "raw" / "detector-compile.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eager": e, "compiled": c,
        "speedup": round(e["mean_ms"] / c["mean_ms"], 3),
        "mean_max_box_diff_px": round(float(np.mean(diffs)), 4) if diffs else None,
    }, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()