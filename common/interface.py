from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class HandPrediction:
    """Normalized output contract every model wrapper must produce."""
    joints_3d: np.ndarray                     # (21, 3) float32, meters, wrist-relative
    confidence: float                         # detector/overall confidence, [0, 1]
    bbox_xyxy: Optional[np.ndarray] = None    # (4,) in source image pixel coords, if available
    mano_pose: Optional[np.ndarray] = None    # (48,) or (45,) axis-angle, if model produces MANO
    mano_shape: Optional[np.ndarray] = None   # (10,) betas, if model produces MANO
    mesh_verts: Optional[np.ndarray] = None   # (778, 3) MANO mesh vertices, if available
    handedness: str = "right"                 # "left" or "right"
    raw: dict = field(default_factory=dict)   # anything model-specific for debugging


class BaseHandModel(ABC):
    """Shared interface. One subclass per candidate model."""

    name: str = "base"

    @abstractmethod
    def load(self, device: str = "cuda") -> None:
        """Load weights, move to device, set eval mode. Called once."""
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, image_bgr: np.ndarray) -> object:
        """Raw OpenCV BGR image -> whatever tensor/batch format infer() expects."""
        raise NotImplementedError

    @abstractmethod
    def infer(self, batch: object) -> list[HandPrediction]:
        """Run the model. Must NOT include preprocessing or postprocessing timing tricks —
        this is the function the benchmark harness times directly."""
        raise NotImplementedError

    def warmup(self, n: int = 20, image_size: tuple[int, int] = (224, 224)) -> None:
        """Run n dummy forward passes. Call before any timed benchmark — first-call CUDA context
        init, cuDNN autotune, and lazy kernel compilation must not pollute your latency numbers."""
        dummy = np.zeros((*image_size, 3), dtype=np.uint8)
        batch = self.preprocess(dummy)
        for _ in range(n):
            self.infer(batch)

    @property
    @abstractmethod
    def device(self) -> str:
        raise NotImplementedError