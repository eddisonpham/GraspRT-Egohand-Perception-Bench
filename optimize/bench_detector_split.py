"""Split profiler for ultralytics detector, exactly mimicking stream_inference.

Hypothesis from torch.compile/imgsz no-ops: detector time is dominated by
Python preprocess + postprocess + transfers, not GPU forward. This measures
each in isolation with the correct input plumbing:
  - preprocess(self.preprocess([im]))   letterbox + normalize
  - forward(det.model(im))              GPU, in-place on the preprocessed batch
  - postprocess(self.postprocess(...))  NMS + Results
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

    loader = FreiHandLoader(subset="data/freihand/subsets/dev.json")
    images = [loader[i][0] for i in range(30)]

    # warmup + init predictor source / model (sets det.predictor)
    for img in images[:5]:
        det(img, verbose=False)
    pred = det.predictor
    assert pred is not None, "predictor not initialized"

    t_pre, t_fwd, t_post = [], [], []
    for img in images:
        # preprocess: takes list of BGR ndarrays
        t0 = time.perf_counter()
        im = pred.preprocess([img])
        t1 = time.perf_counter()
        t_pre.append((t1 - t0) * 1000)

        # forward (GPU) — already on device after preprocess
        t0 = time.perf_counter()
        p = det.model(im, profile=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        t_fwd.append((t1 - t0) * 1000)

        # postprocess: needs the ORIGINAL image as a tensor for NMS
        t0 = time.perf_counter()
        pred.postprocess(p, im, torch.as_tensor(img).permute(2, 0, 1).unsqueeze(0))
        t1 = time.perf_counter()
        t_post.append((t1 - t0) * 1000)

    def summ(arr):
        a = np.asarray(arr)
        return {"mean_ms": round(float(a.mean()), 3),
                "p95_ms": round(float(np.percentile(a, 95)), 3)}

    print("\n=== Detector split (exact stream_inference emulation) ===")
    total = 0
    rows = []
    for name, arr in [("preprocess (letterbox+norm)", t_pre),
                      ("forward (GPU conv)", t_fwd),
                      ("postprocess (NMS+boxes)", t_post)]:
        s = summ(arr)
        total += s["mean_ms"]
        rows.append((name, s))
        print(f"  {name:<28} mean {s['mean_ms']:7.3f}  p95 {s['p95_ms']:7.3f}")
    print(f"  {'sum':<28} mean {total:7.3f}")

    out = ROOT / "results" / "raw" / "detector-split.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "preprocess": summ(t_pre), "forward": summ(t_fwd),
        "postprocess": summ(t_post), "sum_ms": round(total, 3),
    }, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()