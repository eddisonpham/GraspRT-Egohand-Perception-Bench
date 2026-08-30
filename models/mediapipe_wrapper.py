"""Candidate A — MediaPipe Hand Landmarker (floor reference, CPU, no mesh).

Runs in its own conda env (egohand-mediapipe). No GPU path in the standard Tasks API —
expected; this is the deliberately cheap floor per inductive bias #4.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# make `common` importable regardless of how this file is invoked (script or -m)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.interface import BaseHandModel, HandPrediction  # noqa: E402

ASSET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "hand_landmarker.task")


class MediaPipeHandModel(BaseHandModel):
    name = "mediapipe"

    def __init__(self):
        self.model = None
        self._device = "cpu"

    def load(self, device: str = "cuda") -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not os.path.exists(ASSET):
            raise FileNotFoundError(
                f"MediaPipe asset missing at {ASSET} — download hand_landmarker.task first"
            )
        self.base_options = mp_python.BaseOptions(model_asset_path=ASSET)
        self.options = vision.HandLandmarkerOptions(
            base_options=self.base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
        )
        self.model = vision.HandLandmarker.create_from_options(self.options)
        self._device = "cpu"

    def preprocess(self, image_bgr: np.ndarray):
        # wrap BGR ndarray into mp.Image (SRGB) — convert BGR->RGB as documented
        rgb = image_bgr[:, :, ::-1].copy()
        import mediapipe as mp
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    def infer(self, batch) -> list[HandPrediction]:
        res = self.model.detect(batch)
        preds: list[HandPrediction] = []
        if not res.hand_landmarks:
            return preds
        lms = res.hand_world_landmarks[0]  # already 3D, meters, wrist-anchored
        joints = np.asarray([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
        # image-space 2D bbox (approx; use first & last landmark extremes + margin)
        im = res.hand_landmarks[0]
        xs = [p.x for p in im]
        ys = [p.y for p in im]
        h, w = batch.width, batch.height
        x1, y1, x2, y2 = min(xs) * w, min(ys) * h, max(xs) * w, max(ys) * h
        pad = 0.15
        bw, bh = x2 - x1, y2 - y1
        bbox = np.array(
            [max(0, x1 - pad * bw), max(0, y1 - pad * bh),
             min(w, x2 + pad * bw), min(h, y2 + pad * bh)],
            dtype=np.float32,
        )
        handed = "left" if res.handedness[0][0].category_name.lower() == "left" else "right"
        preds.append(HandPrediction(
            joints_3d=joints,
            confidence=float(res.handedness[0][0].score),
            bbox_xyxy=bbox,
            handedness=handed,
            raw={"n_hands": len(res.hand_landmarks)},
        ))
        return preds

    @property
    def device(self) -> str:
        return self._device


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
    from data.freihand.loader import FreiHandLoader

    m = MediaPipeHandModel()
    m.load(device="cpu")
    root = os.environ.get("FREIHAND_ROOT")
    if root:
        L = FreiHandLoader(root=root)
        indices = [0, 1, 2, 3, 4]
    else:
        L = FreiHandLoader(subset="data/freihand/subsets/dev.json")
        indices = list(range(5))
    n_ok = 0
    for i, idx in enumerate(indices):
        # Sparse Windows smoke-root uses its actual file IDs; regular root uses subset index.
        img, gt, K = L[i]
        batch = m.preprocess(img)
        preds = m.infer(batch)
        if preds:
            p = preds[0]
            assert p.joints_3d.shape == (21, 3), p.joints_3d.shape
            assert p.mesh_verts is None
            n_ok += 1
            print(f"img {i}: joints {p.joints_3d.shape}, conf {p.confidence:.2f}, handed {p.handedness}, bbox {p.bbox_xyxy}")
        else:
            print(f"img {i}: no hand detected")
    print(f"detected {n_ok}/5")