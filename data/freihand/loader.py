"""FreiHandLoader — shared, fair data access for every model benchmark.

Root resolution order:
  1. $FREIHAND_ROOT (set explicitly)
  2. ~/egohand_data/freihand   (WSL ext4 fast path — used for actual benchmark runs)
  3. <repo>/data/freihand       (fallback / user-visible copy)

All GT is loaded once at construction (JSON), so per-image access is cheap.
GT units are meters (FreiHAND native). Images are 224x224 BGR jpg (cv2).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np


def resolve_root() -> Path:
    env = os.environ.get("FREIHAND_ROOT")
    if env:
        return Path(env)
    home = Path.home() / "egohand_data" / "freihand"
    if home.exists():
        return home
    here = Path(__file__).resolve().parent
    if (here / "evaluation_xyz.json").exists():
        return here
    raise FileNotFoundError(
        "FreiHAND not found. Set FREIHAND_ROOT or extract into data/freihand/"
    )


class FreiHandLoader:
    def __init__(self, root: str | Path | None = None, subset: str | None = None):
        self.root = Path(root) if root else resolve_root()
        with open(self.root / "evaluation_xyz.json") as f:
            self.joints = np.asarray(json.load(f), dtype=np.float32)
        with open(self.root / "evaluation_K.json") as f:
            self.K = np.asarray(json.load(f), dtype=np.float32)
        with open(self.root / "evaluation_verts.json") as f:
            self.verts = np.asarray(json.load(f), dtype=np.float32)
        self.rgb_dir = self.root / "evaluation" / "rgb"
        self.n = self.joints.shape[0]
        available = sorted(self.rgb_dir.glob("*.jpg"))
        self.available_indices = {int(p.stem) for p in available} if available else None

        if subset is not None:
            sub = Path(subset)
            if not sub.exists():
                sub = Path(__file__).resolve().parent / "subsets" / Path(subset).name
            with open(sub) as f:
                self.indices = json.load(f)
        else:
            self.indices = list(range(self.n))

    def __len__(self):
        return len(self.indices)

    def image_path(self, i: int) -> Path:
        return self.rgb_dir / f"{self.indices[i]:08d}.jpg"

    def __getitem__(self, i: int):
        """Return (image_bgr (224,224,3) uint8, gt_joints (21,3) m, K (3,3))."""
        idx = self.indices[i]
        img = cv2.imread(str(self.image_path(i)), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.image_path(i))
        if img.shape[:2] != (224, 224):
            img = cv2.resize(img, (224, 224))
        return img, self.joints[idx].copy(), self.K[idx].copy()

    def get_gt_verts(self, i: int) -> np.ndarray:
        return self.verts[self.indices[i]].copy()


if __name__ == "__main__":
    L = FreiHandLoader()
    print("samples:", len(L))
    img, j, K = L[0]
    print("image:", img.shape, img.dtype, "| joints:", j.shape, "| K:", K.shape)
    v = L.get_gt_verts(0)
    print("verts:", v.shape, "| sample joint0:", j[0])