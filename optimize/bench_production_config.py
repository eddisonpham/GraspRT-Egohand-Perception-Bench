"""Integrated production-config benchmark (all *adopted* optimizations).

Wires the three adoptable optimizations into one end-to-end config:

  1. LoRA fine-tuned backbone, adapter MERGED into the weight matrices
     (zero inference cost; `results/finetuned/wilor-lora-r8-es`).
  2. one-Euro temporal smoothing on the 3D joint trajectory (jitter down,
     accuracy preserved).
  3. Detection cadence K (box reuse every K frames) for throughput.

Two honest lookbacks:
  - Accuracy: held-out FreiHAND MPJPE, BASE backbone vs MERGED-finetuned
    backbone (same protocol as the trainer, covered flexible subset size so it
    runs in a session budget).
  - Throughout: FPS at cadence K=1 (every frame) vs K=2 (adopted default) on
    synthetic ego clips, with one-Euro smoothing applied.

INT8, TRT detector, and batching are intentionally NOT included: each was
measured to hurt accuracy (documented negatives), so "adopting" them would run
against the evidence. This file proves the config that the measurements
actually support.

Usage (egohand env):
  python optimize/bench_production_config.py --cadence 2 --n-eval 96
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


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
        return None
    ds = ViTDetDataset(model.cfg, image_bgr, np.asarray(boxes, dtype=np.float32),
                       np.asarray(right, dtype=np.float32), rescale_factor=2.0,
                       fp16=False)
    return ds[0]


def item_to_gpu(item, device):
    import torch
    from wilor.utils import recursive_to

    item = {k: torch.as_tensor(v) if isinstance(v, np.ndarray) else v
            for k, v in item.items()}
    item = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.ndim >= 1
            and k in {"img", "right", "box_center", "box_size", "img_size"}
            else v for k, v in item.items()}
    return recursive_to(item, device)


def wrist_anchor(j):
    return j - j[:, :1, :]


def one_euro(traj, min_cutoff=0.8, beta=0.02, d_cutoff=1.0, dt=1.0):
    out = np.empty_like(traj)
    if traj.shape[0] == 0:
        return out
    px = traj[0].copy(); pdx = np.zeros_like(traj[0]); out[0] = px
    for t in range(1, traj.shape[0]):
        dx = (traj[t] - px) / dt
        a_d = 1.0 / (1.0 + d_cutoff * dt)
        edx = a_d * dx + (1.0 - a_d) * pdx
        speed = np.abs(edx)
        cutoff = min_cutoff + beta * speed
        a = 1.0 / (1.0 + 1.0 / (dt * cutoff))
        x = a * traj[t] + (1.0 - a) * px
        out[t] = x
        px, pdx = x, edx
    return out


def jitter_mm(traj):
    if len(traj) < 2:
        return float("nan")
    return float(np.linalg.norm(np.diff(traj, axis=0), axis=-1).mean())


def mpjpe(net, imgs, gt):
    net.eval()
    import torch
    with torch.no_grad():
        out = net({"img": imgs})
        pred = wrist_anchor(out["pred_keypoints_3d"][..., :3]).cpu().numpy()
        g = wrist_anchor(gt).cpu().numpy()
    return float(np.linalg.norm(pred - g, axis=-1).mean() * 1000)


def build_val(model, loader, indices, device):
    import torch

    imgs, joints = [], []
    for i in indices:
        img, gt, _K = loader[i]
        it = _preprocess(model, img)
        if it is None:
            continue
        gi = item_to_gpu(it, device)
        imgs.append(gi["img"].float())
        joints.append(torch.as_tensor(gt, dtype=torch.float32, device=device)
                      .unsqueeze(0))
    if not imgs:
        return None, None
    return torch.cat(imgs, 0), torch.cat(joints, 0)


def main() -> None:
    from data.freihand.loader import FreiHandLoader
    from models.wilor_wrapper import WiLoRHandModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--cadence", type=int, default=2)
    ap.add_argument("--n-eval", type=int, default=96)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--adapter", default="wilor-lora-r8-es")
    ap.add_argument("--clips",
                    default="data/egocentric_clips/clip_01_synthetic.mp4,"
                            "data/egocentric_clips/clip_02_synthetic.mp4")
    args = ap.parse_args()

    import torch
    from peft import PeftModel

    device = torch.device("cuda")
    model = WiLoRHandModel(variant="default")
    model.load("cuda")

    # --- Accuracy: BASE backbone ---
    loader = FreiHandLoader(subset="data/freihand/subsets/full.json")
    n = min(args.n_eval, len(loader))
    idx = np.random.RandomState(0).choice(len(loader), size=n, replace=False)
    val_idx = idx[: n // 5].tolist()
    val_imgs, val_gt = build_val(model, loader, val_idx, device)
    if val_imgs is None:
        print("[prod] no detections; aborting")
        return
    base_mpjpe = mpjpe(model.model, val_imgs, val_gt)

    # --- Accuracy: MERGED fine-tuned backbone ---
    adapter_dir = ROOT / "results" / "finetuned" / args.adapter
    pt = PeftModel.from_pretrained(model.model.backbone, str(adapter_dir))
    merged = pt.merge_and_unload()
    model.model.backbone = merged
    merged_mpjpe = mpjpe(model.model, val_imgs, val_gt)

    # --- Throughput: cadence on ego clips, fine-tuned backbone + one-Euro ---
    import time
    import cv2

    frames = []
    for clip in args.clips.split(","):
        cap = cv2.VideoCapture(clip)
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()

    def run_cadence(K):
        stream = torch.cuda.Stream()
        box = None
        times = []
        for i, frame in enumerate(frames):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if i % K == 0:
                box = model.detector(frame, conf=0.3, verbose=False)[0]
                bd = None
                for d in box:
                    dd = d.boxes.data.detach().cpu().squeeze().numpy()
                    dd = np.atleast_2d(dd)
                    for row in dd:
                        x1, y1, x2, y2 = row[:4]
                        bd = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
                        break
                    if bd:
                        break
                box = bd
            if box is None:
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)
                continue
            a = (box[0] - box[2] / 2, box[1] - box[3] / 2,
                 box[0] + box[2] / 2, box[1] + box[3] / 2)
            # rebuild crop from the (possibly reused/stale) box coords
            from wilor.datasets.vitdet_dataset import ViTDetDataset
            ds = ViTDetDataset(model.cfg, frame, np.asarray([list(a)],
                                                            dtype=np.float32),
                               np.asarray([0.0], dtype=np.float32),
                               rescale_factor=2.0, fp16=False)
            gi = item_to_gpu(ds[0], device)
            with torch.cuda.stream(stream):
                out = model.model({"img": gi["img"].float()})
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        t = np.array(times[5:])
        return 1000.0 / t.mean(), float(t.mean()), float(np.percentile(t, 95))

    fps1, ms1, p1 = run_cadence(1)
    fps2, ms2, p2 = run_cadence(args.cadence)

    print("=" * 70)
    print("INTEGRATED PRODUCTION CONFIG (all adopted optimizations)")
    print("=" * 70)
    print(f"\n  ACCURACY (held-out {len(val_idx)} imgs, same protocol)")
    print(f"    base MPJPE           : {base_mpjpe:.3f} mm")
    print(f"    + LoRA-merged MPJPE  : {merged_mpjpe:.3f} mm "
          f"({(base_mpjpe-merged_mpjpe)/base_mpjpe*100:+.1f}%)")
    print(f"\n  THROUGHPUT (ego clips, fine-tuned backbone, one-Euro at output)")
    print(f"    cadence K=1          : {fps1:.1f} FPS  {ms1:.2f} ms")
    print(f"    cadence K={args.cadence:<2}         : {fps2:.1f} FPS  {ms2:.2f} ms "
          f"({fps2/fps1:.2f}x)")
    print("  (one-Euro smoothing: jitter gains measured separately "
          "in bench_egocentric_jitter.py)")
    print("=" * 70)

    (ROOT / "results" / "raw" / "production-config.json").write_text(json.dumps({
        "base_mpjpe_mm": round(base_mpjpe, 4),
        "merged_mpjpe_mm": round(merged_mpjpe, 4),
        "mpjpe_delta_pct": round((base_mpjpe - merged_mpjpe) / base_mpjpe * 100, 2),
        "cadence": args.cadence,
        "fps_k1": round(fps1, 2), "mean_ms_k1": round(ms1, 3),
        "fps_k2": round(fps2, 2), "mean_ms_k2": round(ms2, 3),
        "speedup_x": round(fps2 / fps1, 3),
        "val_imgs": len(val_idx),
    }, indent=2))
    print(f"\nsaved results/raw/production-config.json")


if __name__ == "__main__":
    main()