# 03 — Repo Scaffold & Shared Interface

Build the actual working project at `./egohand-bench/` (a sibling directory to these instruction
files, not inside them). This keeps the runbook and the deliverable code cleanly separated.

## Directory layout to create

```
egohand-bench/
├── README.md
├── environment.yml                  # or requirements.txt, whichever you used in 01
├── models/
│   ├── __init__.py
│   ├── mediapipe_wrapper.py
│   ├── mobrecon_wrapper.py
│   ├── wilor_wrapper.py
│   └── hamer_wrapper.py
├── common/
│   ├── __init__.py
│   ├── interface.py                 # BaseHandModel ABC + HandPrediction dataclass
│   ├── metrics.py                   # PA-MPJPE, PA-MPVPE, F-scores, procrustes alignment
│   └── profiling.py                 # latency + VRAM measurement utilities
├── data/
│   ├── freihand/                    # populated in 04
│   └── egocentric_clips/            # populated in 04
├── benchmark/
│   ├── run_benchmark.py             # loads one model, runs it over data, writes results/raw/*.json
│   └── aggregate.py                 # builds comparison table + Pareto plot from results/raw/*.json
├── optimize/
│   ├── export_onnx.py
│   ├── run_onnxruntime.py
│   ├── build_tensorrt.py
│   ├── quantize_int8.py
│   └── cuda_graph_capture.py        # optional, stage 10
├── triton/                          # optional, stage 10
│   └── model_repository/
├── tests/
│   └── smoke_test.py                # stage 11
└── results/                         # all raw JSON, plots, notes, FINAL_REPORT.md
    ├── raw/
    ├── plots/
    └── commit_hashes.json
```

## `common/interface.py` — implement this exactly, then never touch it again

Every model wrapper subclasses `BaseHandModel`. The benchmark harness, the ONNX export step, and
the TensorRT step all depend only on this interface — that's what makes swapping/adding models
cheap.

```python
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
```

## `common/metrics.py` — function signatures to implement

```python
def procrustes_align(pred: "np.ndarray", gt: "np.ndarray") -> "np.ndarray":
    """Similarity-transform-align pred (N,3) onto gt (N,3). Returns aligned pred."""

def mpjpe(pred: "np.ndarray", gt: "np.ndarray") -> float:
    """Mean per-joint position error in mm, no alignment."""

def pa_mpjpe(pred: "np.ndarray", gt: "np.ndarray") -> float:
    """Procrustes-Aligned MPJPE in mm. THE headline accuracy metric for this project."""

def pa_mpvpe(pred_verts: "np.ndarray", gt_verts: "np.ndarray") -> float:
    """Procrustes-Aligned mean per-vertex position error in mm. Only for models with mesh_verts."""

def f_score(pred_verts: "np.ndarray", gt_verts: "np.ndarray", threshold_mm: float) -> float:
    """F-score at a distance threshold (5mm and 15mm are the standard FreiHAND thresholds)."""
```

Implement these with standard NumPy/SciPy (`scipy.spatial.procrustes` or a hand-rolled Umeyama
alignment — either is fine, just be consistent across all 4 models).

## Definition of Done

- [ ] `egohand-bench/` directory tree exists matching the layout above.
- [ ] `common/interface.py` is implemented exactly as specified (the ABC contract, not a
      reinterpretation of it) and importable with no errors.
- [ ] `common/metrics.py` has all 5 functions implemented and unit-tested on a trivial case (e.g.
      `pa_mpjpe(gt, gt) == 0.0`).
- [ ] A one-line smoke instantiation test exists proving the ABC correctly rejects an incomplete
      subclass (missing `infer`, for example) — this catches interface drift early.
