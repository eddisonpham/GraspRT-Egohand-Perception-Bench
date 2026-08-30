"""Numeric validation: exported WiLoR ONNX graph vs native PyTorch.

Runs one real FreiHAND crop through (a) native PyTorch forward_step and
(b) ONNX Runtime (CPU + CUDA EP), reporting max abs diff per output and a
PA-MPJPE-style residual after Procrustes alignment. Detector/crop stay outside
the graph by design; validation is crop-to-crop.

Usage (WSL, egohand env):
  WILOR_ROOT=~/src/WiLoR FREIHAND_ROOT=~/egohand_data/freihand \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python optimize/validate_onnx.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_crop(model, freihand_root: str, index: int = 0):
    """Run official preprocess on a real FreiHAND image, return first crop tensors."""
    from data.freihand.loader import FreiHandLoader

    loader = FreiHandLoader(freihand_root)
    img, _gt_joints, _K = loader[index]
    if img is None:
        raise RuntimeError(f"failed to read FreiHAND image #{index}")
    pre = model.preprocess(img)
    ds = pre["dataset"]
    if ds is None:
        raise RuntimeError("detector found no hand on validation image")
    batch = ds[0]
    return batch["img"].unsqueeze(0), batch["right"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(ROOT / "results" / "onnx" / "wilor-fast.onnx"))
    ap.add_argument("--variant", choices=["default", "fast"], default="fast")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--freihand-root", default=os.environ.get("FREIHAND_ROOT", ""))
    vargs = ap.parse_args()
    if not vargs.freihand_root:
        sys.exit("FREIHAND_ROOT not set and --freihand-root not given")

    import onnxruntime as ort
    import torch

    from common.metrics import procrustes_align
    from models.wilor_wrapper import WiLoRHandModel

    model = WiLoRHandModel(variant=vargs.variant)
    model.load("cuda")
    inner = model.model
    eager = getattr(inner.backbone, "_orig_mod", None)
    if eager is not None:
        inner.backbone = eager
    inner.requires_grad_(False)

    img_t, right_t = load_crop(model, vargs.freihand_root, vargs.index)

    # Native PyTorch reference.
    dev = next(inner.parameters()).device
    right_tensor = torch.as_tensor(np.asarray(right_t, dtype=np.float32)).reshape(1)
    with torch.no_grad():
        out = inner.forward_step(
            {"img": img_t.to(dev, dtype=img_t.dtype), "right": right_tensor.to(dev)},
            train=False,
        )
    ref_j = out["pred_keypoints_3d"].float().cpu().numpy()
    ref_v = out["pred_vertices"].float().cpu().numpy()

    # ONNX Runtime sessions. Feed only the inputs the graph actually declares
    # (the traced graph may have baked `right` in as a constant).
    img_np = img_t.cpu().numpy()
    right_np = right_tensor.numpy()
    for providers in (["CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"]):
        try:
            sess = ort.InferenceSession(str(vargs.onnx), providers=providers)
        except Exception as exc:
            print(f"[{providers[0]}] session FAILED: {exc!r}")
            continue
        active = sess.get_providers()
        names = {i.name for i in sess.get_inputs()}
        feed = {"image": img_np}
        if "right" in names:
            feed["right"] = right_np
        oj, ov = sess.run(None, feed)
        dj = np.abs(oj.astype(np.float64) - ref_j.astype(np.float64)).max()
        dv = np.abs(ov.astype(np.float64) - ref_v.astype(np.float64)).max()
        aligned = procrustes_align(oj.astype(np.float64)[0], ref_j.astype(np.float64)[0])
        mpjpe = float(np.linalg.norm(aligned - ref_j[0], axis=-1).mean()) * 1000
        print(f"[{active[0]}] joints max|d|={dj:.3e} verts max|d|={dv:.3e} PA-MPJPE={mpjpe:.3f}mm")
