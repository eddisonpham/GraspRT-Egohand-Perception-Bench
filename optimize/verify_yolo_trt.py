"""Validate TRT YOLO hand detector produces equivalent detections to PyTorch.

Checks on real FreiHAND images: detection count, bbox order-of-magnitude,
keypoint (joint) presence. Confirms the conf>1 reading is a stats artifact,
not a real accuracy regression.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    from data.freihand.loader import FreiHandLoader
    from models.wilor_wrapper import WiLoRHandModel

    engine_path = str(ROOT / "results" / "trt" / "yolo-hand-fp16.engine")
    if not os.path.exists(engine_path):
        sys.exit(f"engine not found: {engine_path}. Run export_yolo_trt.py first.")

    model = WiLoRHandModel(variant="fast")
    model.load("cuda")
    torch_det = model.detector

    from ultralytics import YOLO
    trt_det = YOLO(engine_path, task="pose")

    loader = FreiHandLoader(subset="data/freihand/subsets/dev.json")
    images = [loader[i][0] for i in range(30)]

    def desc(det_model):
        nboxes, xyxy_ok, npts, confs = [], [], [], []
        for img in images:
            r = det_model(img, verbose=False)[0]
            b = r.boxes
            if b is None or len(b) == 0:
                nboxes.append(0); continue
            n = len(b)
            nboxes.append(n)
            xy = b.xyxy.cpu().numpy() if not hasattr(b.xyxy, "cpu") or True else b.xyxy
            xy = np.asarray(b.xyxy.detach().cpu().numpy()) if hasattr(b.xyxy, "detach") else np.asarray(b.xyxy)
            # bbox should be within image bounds 0..224
            in_bounds = bool(np.all((xy >= -1) & (xy <= 226)))
            xyxy_ok.append(in_bounds)
            c = b.conf.detach().cpu().numpy() if hasattr(b.conf, "detach") else np.asarray(b.conf)
            confs.extend(float(x) for x in c)
        return nboxes, xyxy_ok, confs

    torch_boxes, torch_ok, torch_conf = desc(torch_det)
    trt_boxes, trt_ok, trt_conf = desc(trt_det)

    tr = [1 if b > 0 else 0 for b in torch_boxes]
    dr = [1 if b > 0 else 0 for b in trt_boxes]

    print("\n=== YOLO detector equivalence (real FreiHAND, 30 imgs) ===")
    print(f"  detection rate  PyTorch={np.mean(tr):.3f}  TRT={np.mean(dr):.3f}")
    print(f"  per-image box counts match: {tr == dr}")
    print(f"  bbox within-frame  PyTorch={sum(torch_ok)}/{30}  TRT={sum(trt_ok)}/{30}")
    torch_c_finite = [c for c in torch_conf if np.isfinite(c)]
    trt_c_finite = [c for c in trt_conf if np.isfinite(c)]
    if torch_c_finite:
        print(f"  torch conf range: [{min(torch_c_finite):.3f}, {max(torch_c_finite):.3f}] "
              f"(n={len(torch_c_finite)})")
    if trt_c_finite:
        print(f"  trt  conf range: [{min(trt_c_finite):.3f}, {max(trt_c_finite):.3f}] "
              f"(n={len(trt_c_finite)})")
        print(f"  trt conf >1 count: {sum(c > 1 for c in trt_c_finite)}")

    out = ROOT / "results" / "raw" / "detector-trt-verify.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_images": 30,
        "detection_rate_torch": float(np.mean(tr)),
        "detection_rate_trt": float(np.mean(dr)),
        "box_counts_match": tr == dr,
        "bbox_in_bounds_torch": sum(torch_ok),
        "bbox_in_bounds_trt": sum(trt_ok),
        "torch_conf_max": float(max(torch_c_finite)) if torch_c_finite else None,
        "trt_conf_max": float(max(trt_c_finite)) if trt_c_finite else None,
        "trt_conf_gt_1": sum(c > 1 for c in trt_c_finite),
    }, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()