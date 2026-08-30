"""Compare torch vs TRT YOLO detector box coordinates pixel-by-pixel.

Earlier count checks matched, but end-to-end accuracy dropped 5.65->21.4mm.
This checks whether the actual xyxy coordinates differ (the likely cause).
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

    engine_path = os.environ.get(
        "DET_ENGINE", str(ROOT / "results" / "trt" / "yolo-hand-fp32.engine"))
    model = WiLoRHandModel(variant="fast")
    model.load("cuda")
    torch_det = model.detector

    from ultralytics import YOLO
    trt_det = YOLO(engine_path, task="pose")

    loader = FreiHandLoader(subset="data/freihand/subsets/dev.json")
    images = [loader[i][0] for i in range(30)]

    overrides = []  # torch box overrides trt box by this many px
    both_none = 0
    count_agree = 0
    coord_diff = 0.0
    coord_cnt_small = 0
    for img in images:
        # get the single dominant box from each
        def dom(det):
            r = det(img, verbose=False)[0]
            if r.boxes is None or len(r.boxes) == 0:
                return None
            xy = r.boxes.xyxy.detach().cpu().numpy()
            c = r.boxes.conf.detach().cpu().numpy()
            return xy[c.argmax()]
        tb = dom(torch_det)
        db = dom(trt_det)
        if tb is None and db is None:
            both_none += 1
            continue
        if tb is None or db is None:
            coord_diff += 999
            continue
        count_agree += 1
        diff = float(np.abs(tb - db).max())
        coord_diff += diff
        coord_cnt_small += 1

    print("\n=== Box coordinate comparison (torch vs TRT), 30 imgs ===")
    print(f"  both detected: {count_agree}")
    print(f"  both missed:   {both_none}")
    if coord_cnt_small:
        print(f"  mean max-abs box diff: {coord_diff / coord_cnt_small:.3f} px")
        print(f"  (dominant-hand conf-argmax boxes)")

    out = ROOT / "results" / "raw" / "yolo-box-compare.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_images": 30,
        "both_detected": count_agree,
        "both_missed": both_none,
        "mean_max_box_diff_px": round(coord_diff / coord_cnt_small, 3) if coord_cnt_small else None,
    }, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()