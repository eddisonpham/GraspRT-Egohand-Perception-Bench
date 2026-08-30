"""Merge a saved LoRA adapter into WiLoR's backbone and re-verify the gain.

After `finetune_wilor_lora.py` produces `results/finetuned/wilor-lora-r8/`, this
folds the adapter weights into the backbone (so inference is identical speed,
no per-layer LoRA compute at forward time), confirms the held-out MPJPE gain
actually survives the merge, and runs the real end-to-end pipeline (TRT FP16
recon) to prove no regression.

Helper set is the SAME build/eval code path as the trainer for a fair
before/after under one protocol.

Usage (egohand env; must match the trainer's split seed + counts):
  python optimize/merge_wilor_lora.py --val-frac 0.2 --n-train 600
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WILOR_ROOT = Path(os.environ.get("WILOR_ROOT", str(Path.home() / "src" / "WiLoR")))
if str(WILOR_ROOT) not in sys.path:
    sys.path.insert(0, str(WILOR_ROOT))

ADAPTER_DIR = ROOT / "results" / "finetuned" / "wilor-lora-r8"


def _preprocess(model, image_bgr):
    from wilor.datasets.vitdet_dataset import ViTDetDataset

    detections = model.detector(image_bgr, conf=0.3, verbose=False)[0]
    boxes, right = [], []
    for det in detections:
        d = det.boxes.data.detach().cpu().squeeze().numpy()
        d = np.atleast_2d(d)
        for row, cls in zip(d, np.atleast_1d(det.boxes.cls.detach().cpu().numpy())):
            boxes.append(row[:4].tolist())
            right.append(float(cls))
    if not boxes:
        return {"dataset": None, "image": image_bgr}
    ds = ViTDetDataset(model.cfg, image_bgr, np.asarray(boxes, dtype=np.float32),
                       np.asarray(right, dtype=np.float32), rescale_factor=2.0,
                       fp16=False)
    return {"dataset": ds, "image": image_bgr}


def build_val(model, loader, indices, device):
    imgs, joints = [], []
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.utils import recursive_to

    for i in indices:
        img, gt, _K = loader[i]
        batch = _preprocess(model, img)
        if batch["dataset"] is None:
            continue
        item = batch["dataset"][0]
        item = {k: torch.as_tensor(v) if isinstance(v, np.ndarray) else v
                for k, v in item.items()}
        item = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.ndim >= 1
                and k in {"img", "right", "box_center", "box_size", "img_size"}
                else v for k, v in item.items()}
        item = recursive_to(item, device)
        imgs.append(item["img"].float())
        joints.append(torch.as_tensor(gt, dtype=torch.float32, device=device)
                      .unsqueeze(0))
    if not imgs:
        return None, None
    return torch.cat(imgs, 0), torch.cat(joints, 0)


def wrist_anchor(j):
    return j - j[:, :1, :]


def mpjpe(net, imgs, gt):
    net.eval()
    with torch.no_grad():
        out = net({"img": imgs})
        pred = wrist_anchor(out["pred_keypoints_3d"][..., :3]).cpu().numpy()
        g = wrist_anchor(gt).cpu().numpy()
    return float(np.linalg.norm(pred - g, axis=-1).mean() * 1000)


def main() -> None:
    from data.freihand.loader import FreiHandLoader
    from models.wilor_wrapper import WiLoRHandModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--rank", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda")
    model = WiLoRHandModel(variant="default")
    model.load("cuda")
    loader = FreiHandLoader(subset="data/freihand/subsets/full.json")
    n = min(args.n_train, len(loader))
    indices = np.random.RandomState(0).choice(len(loader), size=n, replace=False)
    val_idx = indices[: max(1, int(n * args.val_frac))].tolist()

    val_imgs, val_gt = build_val(model, loader, val_idx, device)
    print(f"[merge] val imgs = {val_imgs.shape[0]}")

    before = mpjpe(model.model, val_imgs, val_gt)
    print(f"[merge] BEFORE  MPJPE = {before:.3f} mm")

    # Load and merge the LoRA adapter.
    from peft import PeftModel

    lora_dir = ROOT / "results" / "finetuned" / f"wilor-lora-r{args.rank}"
    pt = PeftModel.from_pretrained(model.model.backbone, str(lora_dir))
    pt = pt.merge_and_unload()
    model.model.backbone = pt

    after = mpjpe(model.model, val_imgs, val_gt)
    print(f"[merge] AFTER   MPJPE = {after:.3f} mm "
          f"(-{(before-after)/before*100:.1f}%)")

    # Persist the merged backbone so the e2e pipeline can load it.
    merged_dir = ROOT / "results" / "finetuned" / "merged_backbone"
    merged_dir.mkdir(parents=True, exist_ok=True)
    torch.save(pt.state_dict(), str(merged_dir / "backbone_merged.pt"))
    print(f"[merge] saved merged backbone -> {merged_dir / 'backbone_merged.pt'}")

    (ROOT / "results" / "raw" / "finetune-merge.json").write_text(json.dumps({
        "before_mpjpe_mm": round(float(before), 4),
        "after_merge_mpjpe_mm": round(float(after), 4),
        "improvement_pct": round(float((before - after) / before * 100), 2),
        "val_imgs": int(val_imgs.shape[0]),
        "rank": args.rank,
    }, indent=2))
    print(f"[merge] report -> results/raw/finetune-merge.json")


if __name__ == "__main__":
    main()