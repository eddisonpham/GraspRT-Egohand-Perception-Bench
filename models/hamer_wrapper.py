"""Candidate D — HaMeR integration boundary.

HaMeR is deliberately isolated at $HAMER_ROOT. Its official demo requires Detectron2
(ViTDet/RegNetY), ViTPose, a 1.7GB+ demo-data bundle, and MANO. This wrapper exposes the
shared contract and fails explicitly when those official runtime gates are absent; it never
substitutes synthetic predictions for a missing accuracy benchmark.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.interface import BaseHandModel, HandPrediction  # noqa: E402


class HamerHandModel(BaseHandModel):
    name = "hamer"

    def __init__(self, root: str | None = None):
        self.root = Path(root or os.environ.get("HAMER_ROOT", str(Path.home() / "src" / "hamer")))
        self._device = "cpu"

    def load(self, device: str = "cuda") -> None:
        missing = []
        if not (self.root / "hamer" / "models").exists():
            missing.append(f"HaMeR source at {self.root}")
        if not (self.root / "_DATA" / "data" / "MANO_RIGHT.pkl").exists():
            missing.append(f"MANO_RIGHT.pkl at {self.root / '_DATA/data'}")
        for name in ["detectron2", "vitpose_model"]:
            try:
                __import__(name)
            except ImportError:
                missing.append(name)
        checkpoint = Path(os.environ.get("HAMER_CHECKPOINT", str(Path.home() / ".cache" / "hamer" / "hamer_ckpts" / "checkpoints" / "hamer.ckpt")))
        if not checkpoint.exists():
            missing.append(f"HaMeR checkpoint at {checkpoint}")
        if missing:
            raise RuntimeError(
                "HaMeR cannot be benchmarked under the current environment; missing official "
                "runtime gates: " + "; ".join(missing)
            )
        raise NotImplementedError(
            "HaMeR detector+ViTPose integration requires its official demo pipeline; no result "
            "is emitted until that pipeline is validated end-to-end."
        )

    def preprocess(self, image_bgr: np.ndarray):
        return image_bgr

    def infer(self, batch) -> list[HandPrediction]:
        raise RuntimeError("HaMeR is not loaded")

    @property
    def device(self) -> str:
        return self._device