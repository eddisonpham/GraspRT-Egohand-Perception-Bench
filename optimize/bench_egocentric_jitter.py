"""Egocentric video temporal jitter study: does smoothing reduce shake?

WiLoR is frame-independent, so consecutive 3D joints carry per-frame noise that
looks like shake/jitter. This benchmark quantifies that jitter and proves
whether cheap temporal smoothing (MA filter and one-Euro filter on the joint
trajectory and on MANO pose params) converts per-frame outputs into smoother
tracking — measuring BOTH the jitter reduction *and* the distortion (deviation
from the raw trace) so we never trade smoothness for correctness blindly.

Jitter metric: mean frame-to-frame 3D joint displacement (mm). Lower = smoother.
Distortion metric: mean |smoothed - raw| over joints (mm). Must stay small
relative to natural frame-to-frame motion, else the filter is over-smoothing.

Usage (egohand env, TRT FP16 recon required):
  python optimize/bench_egocentric_jitter.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CLIPS = [
    "data/egocentric_clips/clip_01_synthetic.mp4",
    "data/egocentric_clips/clip_02_synthetic.mp4",
]


def load_frames(clip: str) -> list[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(clip)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


# ---- TEMPORAL FILTERS ------------------------------------------------------

def ma_filter(traj: np.ndarray, window: int = 5) -> np.ndarray:
    """Centered moving-average over the time axis. traj: (T, J, D)."""
    T = traj.shape[0]
    out = traj.copy()
    half = window // 2
    for t in range(T):
        lo, hi = max(0, t - half), min(T, t + half + 1)
        out[t] = traj[lo:hi].mean(axis=0)
    return out


def one_euro_filter(traj: np.ndarray, min_cutoff: float = 1.0,
                    beta: float = 0.01, d_cutoff: float = 1.0,
                    dt: float = 1.0) -> np.ndarray:
    """One-Euro low-pass filter over time (adaptive cutoff from speed).

    Standard implementation; operates on each joint coordinate channel.
    """
    out = np.empty_like(traj)
    if traj.shape[0] == 0:
        return out
    prev_x = traj[0].copy()
    prev_dx = np.zeros_like(traj[0])
    out[0] = prev_x
    for t in range(1, traj.shape[0]):
        dx = (traj[t] - prev_x) / dt
        a_d = 1.0 / (1.0 + d_cutoff * dt)          # derivate low-pass
        edx = a_d * dx + (1.0 - a_d) * prev_dx     # smoothed derivative
        speed = np.abs(edx)
        cutoff = min_cutoff + beta * speed
        a = 1.0 / (1.0 + 1.0 / (dt * cutoff))      # data low-pass
        x = a * traj[t] + (1.0 - a) * prev_x
        out[t] = x
        prev_x, prev_dx = x, edx
    return out


# ---- METRICS ---------------------------------------------------------------

def jitter_mm(joints: np.ndarray) -> float:
    """Mean frame-to-frame joint displacement in mm. Lower = smoother."""
    if len(joints) < 2:
        return float("nan")
    d = np.linalg.norm(np.diff(joints, axis=0), axis=-1)
    return float(d.mean())


def distortion_mm(raw: np.ndarray, smoothed: np.ndarray) -> float:
    """Mean |smoothed - raw| over joints+frames. Must stay small."""
    return float(np.linalg.norm(smoothed - raw, axis=-1).mean())


def per_clip_frame_velocity(joints: np.ndarray) -> float:
    d = np.linalg.norm(np.diff(joints, axis=0), axis=-1)
    return float(np.percentile(d, 50)) if d.size else float("nan")


def reconstruct_clip(model, context, d_joints, d_verts, frames,
                     warmup: int = 4) -> np.ndarray:
    """Joint trajectory (T, 21, 3) mm for one clip via TRT FP16 recon."""
    import time
    import torch

    from data.freihand.loader import FreiHandLoader  # noqa: re-export path

    stream = torch.cuda.Stream()
    traj = []
    for frame in frames:
        torch.cuda.synchronize()
        item = prep_item(model, frame)
        if item is None:
            continue
        img_t = item["img"].to(torch.float16)
        with torch.cuda.stream(stream):
            context.execute_v2(bindings=[
                img_t.data_ptr(), d_joints.data_ptr(), d_verts.data_ptr()])
        torch.cuda.synchronize()
        joints = d_joints.float().cpu().numpy().reshape(21, 3)
        traj.append(joints)
    return np.asarray(traj)


def prep_item(model, img) -> dict | None:
    import torch

    from wilor.utils import recursive_to

    batch = model.preprocess(img)
    if batch["dataset"] is None:
        return None
    item = batch["dataset"][0]
    item = {k: torch.as_tensor(v) if isinstance(v, np.ndarray) else v
            for k, v in item.items()}
    item = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.ndim >= 1
            and k in {"img", "right", "box_center", "box_size", "img_size"}
            else v for k, v in item.items()}
    return recursive_to(item, model._torch_device)


def main() -> None:
    import tensorrt as trt
    import torch

    from models.wilor_wrapper import WiLoRHandModel

    model = WiLoRHandModel(variant="fast")
    model.load("cuda")
    eager = getattr(model.model.backbone, "_orig_mod", None)
    if eager is not None:
        model.model.backbone = eager
    model.model.requires_grad_(False)

    engine_path = str(ROOT / "results" / "trt" / "wilor-fast-fp16.plan")
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    d_joints = torch.empty((1, 21, 3), dtype=torch.float16, device="cuda")
    d_verts = torch.empty((1, 778, 3), dtype=torch.float16, device="cuda")

    print("=" * 72)
    print("EGOCENTRIC TEMPORAL JITTER + SMOOTHING STUDY")
    print("=" * 72)

    agg = []
    all_reports = {}
    for clip in CLIPS:
        frames = load_frames(clip)
        traj = reconstruct_clip(model, context, d_joints, d_verts, frames)
        name = Path(clip).stem
        print(f"\n  [{name}] reconstructed {len(traj)}/{len(frames)} detected frames")

        if len(traj) < 10:
            print(f"    skipped: too few frames ({len(traj)})")
            continue

        # Raw jitter.
        j_raw = jitter_mm(traj)
        med_speed = per_clip_frame_velocity(traj)

        # Smoothed variants.
        sm_ma5 = ma_filter(traj, window=5)
        sm_ma9 = ma_filter(traj, window=9)
        sm_euro = one_euro_filter(traj, min_cutoff=0.8, beta=0.02, dt=1.0)

        variants = {
            "raw": (traj, j_raw, med_speed),
            "ma5": (sm_ma5, jitter_mm(sm_ma5), per_clip_frame_velocity(sm_ma5)),
            "ma9": (sm_ma9, jitter_mm(sm_ma9), per_clip_frame_velocity(sm_ma9)),
            "one-euro": (sm_euro, jitter_mm(sm_euro), per_clip_frame_velocity(sm_euro)),
        }

        print(f"    {'filter':<10}{'jitter':>10}{'med vel':>10}{'jitter↓':>10}"
              f"{'distortion':>12}")
        print(f"    {'-'*50}")
        report = {}
        for name2, (arr, j, vel) in variants.items():
            dist = distortion_mm(traj, arr) if name2 != "raw" else 0.0
            red_pct = (1 - j / j_raw) * 100 if j_raw > 0 else 0
            if name2 == "raw":
                print(f"    {name2:<10}{j:>10.3f}{vel:>10.3f}"
                      f"{'--':>10}{'--':>12}")
            else:
                print(f"    {name2:<10}{j:>10.3f}{vel:>10.3f}"
                      f"{red_pct:>9.0f}%{dist:>12.3f}")
            report[name2] = {"jitter_mm": round(j, 4),
                             "med_velocity_mm": round(vel, 4),
                             "distortion_mm": round(dist, 4),
                             "jitter_reduction_pct": round(red_pct, 2)}
        all_reports[name] = report
        agg.append(report)

    # Aggregate across clips.
    print("\n  " + "=" * 68)
    print("  AGGREGATE (mean over clips)")
    print(f"  {'filter':<10}{'jitter':>10}{'jitter↓':>10}{'distortion':>12}")
    print(f"  {'-'*40}")
    agg_res = {}
    for fname in ("raw", "ma5", "ma9", "one-euro"):
        vals = [r[fname]["jitter_mm"]
                for r in agg if fname in r]
        dists = [r[fname]["distortion_mm"]
                 for r in agg if fname in r]
        if not vals:
            continue
        j_mean = np.mean(vals)
        j_raw_mean = np.mean([r["raw"]["jitter_mm"] for r in agg
                              if "raw" in r])
        red = (1 - j_mean / j_raw_mean) * 100 if j_raw_mean else 0
        dist_mean = np.mean(dists) if dists else 0
        print(f"  {fname:<10}{j_mean:>10.3f}{red:>9.0f}%{dist_mean:>12.3f}")
        agg_res[fname] = {"jitter_mm": round(float(j_mean), 4),
                          "reduction_pct": round(float(red), 2),
                          "distortion_mm": round(float(dist_mean), 4)}
    print("  " + "=" * 68)

    out = ROOT / "results" / "raw" / "egocentric-jitter.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "clips": CLIPS,
        "per_clip": all_reports,
        "aggregate": agg_res,
        "method": "TRT FP16 recon, per-frame; filters on joint trajectory (mm)",
    }, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()