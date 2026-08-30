"""Candidate C — official WiLoR adapter.

The official repository remains external at $WILOR_ROOT (default ~/src/WiLoR) because
its dependency tree is intentionally isolated. This wrapper mirrors demo.py exactly:
YOLO hand detector -> ViTDetDataset crop/normalization -> WiLoR MANO reconstruction.

Required assets in $WILOR_ROOT:
  pretrained_models/{detector.pt,wilor_final.ckpt,model_config.yaml}
  mano_data/MANO_RIGHT.pkl (+ mano_mean_params.npz, supplied by repo)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# make project-local common importable when invoked as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.interface import BaseHandModel, HandPrediction  # noqa: E402


class WiLoRHandModel(BaseHandModel):
    name = "wilor"

    def __init__(self, variant: str = "default", root: str | None = None):
        self.variant = variant
        self.root = Path(root or os.environ.get("WILOR_ROOT", str(Path.home() / "src" / "WiLoR")))
        self.model = None
        self.detector = None
        self.cfg = None
        self._device = "cpu"
        self._imports_ready = False

    def _prepare_imports(self):
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        self._imports_ready = True

    def load(self, device: str = "cuda") -> None:
        self._prepare_imports()
        import torch
        from ultralytics import YOLO
        from wilor.models import load_wilor

        dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        ckpt = self.root / "pretrained_models" / "wilor_final.ckpt"
        cfg = self.root / "pretrained_models" / "model_config.yaml"
        detector_path = self.root / "pretrained_models" / "detector.pt"
        for p in (ckpt, cfg, detector_path, self.root / "mano_data" / "MANO_RIGHT.pkl"):
            if not p.exists():
                raise FileNotFoundError(f"WiLoR asset missing: {p}")
        # load_wilor uses relative MANO paths in its current implementation
        old = os.getcwd()
        os.chdir(self.root)
        try:
            self.model, self.cfg = load_wilor(str(ckpt), str(cfg))
            self.detector = YOLO(str(detector_path))
        finally:
            os.chdir(old)
        if self.variant == "fast":
            # Exact official demo.py fast mode: FP16 + torch.compile backbone + skip blocks.
            torch.set_float32_matmul_precision("high")
            self.model = self.model.half()
            try:
                self.model.backbone = torch.compile(self.model.backbone)
            except Exception:
                # Compilation can fail on unsupported Blackwell/custom-op combinations;
                # FP16 remains valid and is explicitly recorded in raw notes.
                pass
            self.model.backbone.skip_blocks = True
        self.model = self.model.to(dev).eval()
        self.detector = self.detector.to(dev)
        self._device = "cuda" if dev.type == "cuda" else "cpu"
        self._torch_device = dev

    def preprocess(self, image_bgr: np.ndarray):
        # Detection runs here so infer() only receives official crop dataset state.
        from wilor.datasets.vitdet_dataset import ViTDetDataset
        detections = self.detector(image_bgr, conf=0.3, verbose=False)[0]
        boxes, right = [], []
        for det in detections:
            d = det.boxes.data.detach().cpu().squeeze().numpy()
            # YOLO can return a scalar after squeeze for exactly one detection; normalize.
            d = np.atleast_2d(d)
            for row, cls in zip(d, np.atleast_1d(det.boxes.cls.detach().cpu().numpy())):
                boxes.append(row[:4].tolist())
                right.append(float(cls))
        if not boxes:
            return {"dataset": None, "image": image_bgr}
        ds = ViTDetDataset(
            self.cfg, image_bgr, np.asarray(boxes, dtype=np.float32),
            np.asarray(right, dtype=np.float32), rescale_factor=2.0,
            fp16=self.variant == "fast",
        )
        return {"dataset": ds, "image": image_bgr}

    def infer(self, batch) -> list[HandPrediction]:
        if batch["dataset"] is None:
            return []
        import torch
        from wilor.utils import recursive_to

        # benchmark protocol is one dominant hand; use first detection to keep batch=1 VRAM bounded
        item = batch["dataset"][0]
        item = {k: torch.as_tensor(v) if isinstance(v, np.ndarray) else v for k, v in item.items()}
        item = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.ndim >= 1 and k in {"img", "right", "box_center", "box_size", "img_size"} else v for k, v in item.items()}
        item = recursive_to(item, self._torch_device)
        with torch.no_grad():
            out = self.model(item)
        joints = out["pred_keypoints_3d"][0].detach().float().cpu().numpy()
        verts = out["pred_vertices"][0].detach().float().cpu().numpy()
        pose = out["pred_mano_params"]["hand_pose"][0].detach().float().cpu().numpy().reshape(-1)
        betas = out["pred_mano_params"]["betas"][0].detach().float().cpu().numpy().reshape(-1)
        # Official output is right/left normalized; preserve its convention and anchor at wrist.
        joints = joints - joints[0:1]
        verts = verts - joints[0:1]
        right_value = item["right"]
        if hasattr(right_value, "detach"):
            right_value = right_value.detach().cpu().reshape(-1)[0].item()
        else:
            right_value = np.asarray(right_value).reshape(-1)[0].item()
        right = bool(right_value)
        return [HandPrediction(
            joints_3d=joints.astype(np.float32),
            confidence=1.0,
            mano_pose=pose.astype(np.float32),
            mano_shape=betas.astype(np.float32),
            mesh_verts=verts.astype(np.float32),
            handedness="right" if right else "left",
            raw={"variant": self.variant},
        )]

    @property
    def device(self) -> str:
        return self._device


if __name__ == "__main__":
    import cv2
    from data.freihand.loader import FreiHandLoader
    m = WiLoRHandModel(variant=os.environ.get("WILOR_VARIANT", "default"))
    m.load("cuda")
    L = FreiHandLoader(subset="data/freihand/subsets/dev.json")
    img, _, _ = L[0]
    p = m.infer(m.preprocess(img))
    print("predictions:", len(p))
    if p:
        print(p[0].joints_3d.shape, p[0].mesh_verts.shape)
