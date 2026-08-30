"""LoRA fine-tune WiLoR's ViT backbone on real FreiHAND 3D-joint ground truth.

Purpose: demonstrate that the fine-tuning machinery is real and that a low-rank
adaptation moves reconstruction accuracy — measured honestly by a held-out
subset (before vs after MPJPE), not by training-loss curves.

Pipeline-faithful training: every crop is produced by the *same* detector ->
ViTDetDataset -> normalize path used at inference, so the model learns the input
distribution it actually sees. GT joints are wrist-anchored meters (FreiHAND
native). Loss is L1 regression on predicted vs GT 3D joints, matching the
Keypoint3DLoss convention.

Only LoRA adapters on the backbone attention/MLP (qkv, proj, fc1, fc2) are
trainable; everything else is frozen. Fits 8.5GB VRAM (RTX 5060) without
gradient checkpointing at batch 1-2.

Usage (egohand env; WILOR_ROOT must be set):
  python optimize/finetune_wilor_lora.py --epochs 3 --val-frac 0.2
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WILOR_ROOT = Path(os.environ.get("WILOR_ROOT", str(Path.home() / "src" / "WiLoR")))
if str(WILOR_ROOT) not in sys.path:
    sys.path.insert(0, str(WILOR_ROOT))


def build_crops(model, loader, indices, device) -> tuple:
    """Pipeline-faithful crops + GT for the given indices.

    Returns (imgs tensor [N,3,256,256] fp32 normalized, gt_joints [N,21,3] m).
    Skips indices where the detector finds no hand.
    """
    import torch

    from wilor.datasets.vitdet_dataset import ViTDetDataset

    imgs, joints = [], []
    ok_ids = []
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
        from wilor.utils import recursive_to
        item = recursive_to(item, device)
        imgs.append(item["img"].float())          # [1,3,256,256]
        joints.append(torch.as_tensor(gt, dtype=torch.float32, device=device)
                      .unsqueeze(0))
        ok_ids.append(i)
    if not imgs:
        return None, None, []
    return torch.cat(imgs, 0), torch.cat(joints, 0), ok_ids


def _preprocess(model, image_bgr):
    """Mirror model.preprocess but keep on device-free (returns dataset state)."""
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


def wrist_anchor(joints: torch.Tensor) -> torch.Tensor:
    return joints - joints[:, :1, :]


def eval_mpjpe(net, imgs, gt, device) -> float:
    """PA-free MPJPE (mm) on wrist-anchored joints. Smaller = better."""
    import torch

    was_training = net.training
    net.eval()
    with torch.no_grad():
        out = net({"img": imgs})
        pred = out["pred_keypoints_3d"][..., :3]
    pa_pred = wrist_anchor(pred).cpu().numpy()
    pa_gt = wrist_anchor(gt).cpu().numpy()
    if was_training:
        net.train()
    return float(np.linalg.norm(pa_pred - pa_gt, axis=-1).mean() * 1000)


def main() -> None:
    from peft import LoraConfig, get_peft_model

    import torch
    from torch import nn
    from data.freihand.loader import FreiHandLoader
    from models.wilor_wrapper import WiLoRHandModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--patience", type=int, default=3,
                    help="early-stop after this many epochs without val improvement")
    ap.add_argument("--lr-patience", type=int, default=2,
                    help="ReduceLROnPlateau patience (epochs)")
    ap.add_argument("--lr-factor", type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def emit(s):
        print(s, flush=True)

    emit(f"[finetune] device={device}")

    # Load WiLoR default (fp32) with the real asset path.
    model = WiLoRHandModel(variant="default")
    model.load("cuda")

    # Wrap the backbone in LoRA, then freeze every non-LoRA param so only the
    # adapters train (avoid freezing the adapters by a blanket requires_grad_).
    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        target_modules=["qkv", "proj", "fc1", "fc2"],
        lora_dropout=0.05,
        bias="none",
        task_type=None,
    )
    net = get_peft_model(model.model.backbone, lora_cfg)
    net.print_trainable_parameters()
    model.model.backbone = net
    for name, p in model.model.named_parameters():
        if "lora_" not in name:
            p.requires_grad_(False)
    trainable = [p for p in model.model.parameters() if p.requires_grad]
    print(f"[finetune] trainable params: {sum(p.numel() for p in trainable):,}")

    loader = FreiHandLoader(subset="data/freihand/subsets/full.json")
    n = args.n_train
    if n > len(loader):
        n = len(loader)
    indices = np.random.RandomState(0).choice(len(loader), size=n, replace=False)
    val_n = max(1, int(n * args.val_frac))
    train_idx = indices[val_n:].tolist()
    val_idx = indices[:val_n].tolist()
    print(f"[finetune] train={len(train_idx)} val={len(val_idx)}")

    # Build crops.
    train_imgs, train_gt, ok_tr = build_crops(model, loader, train_idx, device)
    val_imgs, val_gt, _ok_va = build_crops(model, loader, val_idx, device)
    if val_imgs is None or train_imgs is None:
        print("[finetune] no detections; aborting")
        return

    torch.cuda.reset_peak_memory_stats()
    before = eval_mpjpe(model.model, val_imgs, val_gt, device)
    print(f"\n[finetune] BEFORE (val {val_imgs.shape[0]} imgs) MPJPE = {before:.3f} mm")
    vram0 = torch.cuda.max_memory_allocated() / 1e6

    # Optimizer only over LoRA adapters; ReduceLROnPlateau on val MPJPE.
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=args.lr_factor, patience=args.lr_patience)
    loss_fn = nn.L1Loss(reduction="mean")

    model.model.train()
    n_tr = train_imgs.shape[0]
    best = before
    best_epoch = -1
    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    val_hist = []
    epochs_run = 0
    no_improve = 0
    import json

    for ep in range(args.epochs):
        epochs_run += 1
        perm = torch.randperm(n_tr)
        tot = 0.0; nbatch = 0
        for s in range(0, n_tr, args.batch):
            idxb = perm[s:s + args.batch]
            x = train_imgs[idxb]
            y = wrist_anchor(train_gt[idxb])
            opt.zero_grad(set_to_none=True)
            out = model.model({"img": x})
            pred = wrist_anchor(out["pred_keypoints_3d"][..., :3])
            loss = loss_fn(pred.float(), y.float())
            loss.backward()
            opt.step()
            tot += loss.item(); nbatch += 1
        avg = tot / max(1, nbatch)
        val = eval_mpjpe(model.model, val_imgs, val_gt, device)
        sched.step(val)
        val_hist.append(round(float(val), 4))
        lr = opt.param_groups[0]["lr"]
        print(f"[finetune] epoch {ep}: train loss={avg:.5f} | "
              f"val MPJPE={val:.3f} mm | lr={lr:.1e}")
        if val < best - 1e-6:
            best = val
            best_state = {k: v.detach().clone()
                          for k, v in net.state_dict().items()}
            no_improve = 0
            best_epoch = ep
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[finetune] early stop at epoch {ep} "
                      f"(best {best:.3f} at ep {best_epoch})")
                break
        model.model.train()

    # Restore best state so the persisted adapter = best val-checkpoint.
    net.load_state_dict(best_state)
    best = eval_mpjpe(model.model, val_imgs, val_gt, device)

    vram_peak = torch.cuda.max_memory_allocated() / 1e6
    print(f"\n[finetune] RESULT: before={before:.3f}mm after={best:.3f}mm "
          f"(-{(before-best)/before*100:.1f}%)")
    print(f"[finetune] epochs_run={epochs_run} | VRAM peak (train)={vram_peak:.1f} MB")

    # Persist the BEST LoRA adapter so it can be merged back at inference.
    out_dir = ROOT / "results" / "finetuned"
    out_dir.mkdir(parents=True, exist_ok=True)
    net.save_pretrained(str(out_dir / f"wilor-lora-r{args.rank}-es"))
    np.save(out_dir / "scores-es.npy",
            np.array([before, best]))
    print(f"[finetune] saved best adapter -> "
          f"{out_dir / f'wilor-lora-r{args.rank}-es'}")

    report = {"before_mpjpe_mm": round(float(before), 4),
              "after_mpjpe_mm": round(float(best), 4),
              "improvement_pct": round(float((before - best) / before * 100), 2),
              "epochs_run": int(epochs_run),
              "patience": args.patience,
              "best_epoch": int(best_epoch),
              "val_history_mm": val_hist,
              "train_imgs": int(n_tr), "val_imgs": int(val_imgs.shape[0]),
              "rank": args.rank, "lr": args.lr,
              "vram_peak_mb": round(float(vram_peak), 1)}
    (ROOT / "results" / "raw" / "finetune-lora.json").write_text(
        json.dumps(report, indent=2))
    print(f"[finetune] report (with hist + early-stop) -> "
          f"results/raw/finetune-lora.json")


if __name__ == "__main__":
    main()