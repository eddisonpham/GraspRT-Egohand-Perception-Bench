"""Inspect the ultralytics YOLO-pose detector structure to find the backbone/head split."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    from models.wilor_wrapper import WiLoRHandModel

    m = WiLoRHandModel(variant="fast")
    m.load("cuda")
    det = m.detector

    print("det.task:", getattr(det, "task", None))
    print("model type:", type(det.model).__name__)
    print("model attrs:", [a for a in dir(det.model) if not a.startswith("_")][:25])

    # ultralytics DetectionModel: .model is the nn.Sequential
    seq = det.model.model
    print("\nseq len:", len(seq))
    for i, mod in enumerate(seq):
        t = type(mod).__name__
        print(f"  [{i}] {t}")

    # Check if there's a split into backbone / head (YOLOv8 has detect head w/ m = self.model)
    if hasattr(det.model, "detect"):
        d = det.model.detect
        print("\ndetect head:", type(d).__name__)
        print("  has stride:", hasattr(d, "stride")) 
        print("  cv2/cv3 (pose) present:", hasattr(d, "cv2") and hasattr(d, "cv3"))


if __name__ == "__main__":
    main()