"""Shared model interface: every wrapper returns a HandPrediction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class HandPrediction:
    """Normalized output contract every model wrapper must produce."""

    joints_3d: np.ndarray          # (21, 3) float32, meters, wrist-relative
    confidence: float              # detector/overall confidence in [0, 1]
    bbox_xyxy: Optional[np.ndarray] = None   # (4,) source-image pixels
    mano_pose: Optional[np.ndarray] = None   # (48,) or (45,) axis-angle
    mano_shape: Optional[np.ndarray] = None  # (10,) betas
    mesh_verts: Optional[np.ndarray] = None  # (778, 3) MANO mesh, meters
    handedness: str = "right"
    raw: dict = field(default_factory=dict)


class BaseHandModel(ABC):
    """One subclass per candidate model."""

    name: str = "base"

    @abstractmethod
    def load(self, device: str = "cuda") -> None:
        """Load weights, move to device, set eval mode."""

    @abstractmethod
    def preprocess(self, image_bgr: np.ndarray) -> object:
        """Raw BGR image -> tensor/batch format for infer()."""

    @abstractmethod
    def infer(self, batch: object) -> list[HandPrediction]:
        """Run the model; timed directly by the benchmark harness."""

    def warmup(self, n: int = 20, image_size: tuple[int, int] = (224, 224)) -> None:
        """Run dummy forward passes to warm CUDA/cuDNN before timing."""
        dummy = np.zeros((*image_size, 3), dtype=np.uint8)
        batch = self.preprocess(dummy)
        for _ in range(n):
            self.infer(batch)

    @property
    @abstractmethod
    def device(self) -> str:
        pass