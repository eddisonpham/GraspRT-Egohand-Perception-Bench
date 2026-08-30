"""Detection-cadence box reuse on egocentric video — the untested lever.

All prior detector experiments tried to make the forward (13.8ms) faster
per-frame; the GPU wall blocks that. But in video the hand box is temporally
correlated: the detector needs to run only every K-th frame and the box is
reused+cropped for the (K-1) intermediates. This amortizes the detector cost
without touching reconstruct accuracy *if* consecutive boxes are stable.

Two honest checks here:
  1. Box-stability on real synthetic ego clips: median/max box displacement
     across consecutive frames. If small, cadence is accuracy-safe.
  2. Throughput: e2e FPS with detector cadence K=1,2,3,4 on the same 224x224
     clips, TRT FP16 recon each frame, serialized pipeline.

This is a throughput demo; the synthetic clips have no FreiHAND ground truth,
so accuracy is reported as *box drift* (the exact proxy that proved sensitive
earlier), not PA-MPJPE.

Usage (WSL, egohand env):
  WILOR_ROOT=~/src/WiLoR TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    python optimize/bench_box_cadence.py
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


def detect_boxes(model, frame):
    """Return (center_xy, size_xy) for the dominant detection, else None."""
    det = model.detector(frame, conf=0.3, verbose=False)[0]
    boxes = []
    for det in det:
        d = det.boxes.data.detach().cpu().squeeze().numpy()
        d = np.atleast_2d(d)
        for row in d:
            x1, y1, x2, y2 = row[:4]
            boxes.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1))
    if not boxes:
        return None
    # dominant = largest area
    return max(boxes, key=lambda b: b[2] * b[3])


def box_stability(model, frames) -> dict:
    """Per-frame detection, then displacement stats between consecutive frames."""
    boxes = [detect_boxes(model, f) for f in frames]
    old = None
    disps = []
    skipped = 0  # frames with no detection
    for b in boxes:
        if b is None:
            skipped += 1
            continue
        if old is not None:
            disps.append([abs(b[0] - old[0]), abs(b[1] - old[1]),
                          abs(b[2] - old[2]), abs(b[3] - old[3])])
        old = b
    d = np.asarray(disps) if disps else np.zeros((0, 4))

    def row(i):
        return d[:, i] if d.shape[0] else np.array([0.0])

    return {
        "n_frames": len(frames),
        "n_detected": sum(b is not None for b in boxes),
        "skipped_no_det": skipped,
        "center_dx_px_median": float(np.median(row(0))),
        "center_dy_px_median": float(np.median(row(1))),
        "center_dx_px_p95": float(np.percentile(row(0), 95)) if d.shape[0] else 0.0,
        "center_dy_px_p95": float(np.percentile(row(1), 95)) if d.shape[0] else 0.0,
        "center_dx_px_max": float(row(0).max()) if d.shape[0] else 0.0,
        "center_dy_px_max": float(row(1).max()) if d.shape[0] else 0.0,
        "size_dx_px_median": float(np.median(row(2))),
        "size_dy_px_median": float(np.median(row(3))),
    }


def prep_from_box(model, frame, box) -> dict | None:
    """Crop using a box (own or a reused/stale one). GPU tensor out."""
    import torch

    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.utils import recursive_to

    if box is None:
        return None
    cx, cy, w, h = box
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    boxes = np.asarray([[x1, y1, x2, y2]], dtype=np.float32)
    right = np.asarray([0.0], dtype=np.float32)
    ds = ViTDetDataset(model.cfg, frame, boxes, right, rescale_factor=2.0,
                       fp16=model.variant == "fast")
    item = ds[0]
    item = {k: torch.as_tensor(v) if isinstance(v, np.ndarray) else v
            for k, v in item.items()}
    item = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.ndim >= 1
            and k in {"img", "right", "box_center", "box_size", "img_size"}
            else v for k, v in item.items()}
    return recursive_to(item, model._torch_device)


def bench_cadence(model, context, d_joints, d_verts, frames, K: int,
                  warmup: int = 5) -> tuple[float, float, float]:
    """Run serialized single-frame pipeline with detection every K frames.

    detector runs every K-th frame; the (K-1) intermediates reuse the last box.
    Returns (FPS, mean_ms, p95_ms) over full frames minus warmup.
    """
    import time
    import torch

    stream = torch.cuda.Stream()
    box = None
    times = []
    for i, frame in enumerate(frames):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if i % K == 0:
            box = detect_boxes(model, frame)
        item = prep_from_box(model, frame, box)
        if item is None:
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
            continue
        img_t = item["img"].to(torch.float16)
        with torch.cuda.stream(stream):
            context.execute_v2(bindings=[
                img_t.data_ptr(), d_joints.data_ptr(), d_verts.data_ptr()])
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    t = np.array(times[warmup:])
    return 1000.0 / t.mean(), float(t.mean()), float(np.percentile(t, 95))


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

    all_frames = []
    for clip in CLIPS:
        all_frames.extend(load_frames(clip))

    print("=" * 70)
    print("DETECTION-CADENCE BOX REUSE (synthetic ego clips, TRT FP16 recon)")
    print("=" * 70)

    # Pass 1: box stability (detector every frame).
    stability = box_stability(model, all_frames)
    print(f"\n  Frames:                {stability['n_frames']}")
    print(f"  Detected:              {stability['n_detected']}"
          f"/{stability['n_frames']} ({stability['n_detected']/stability['n_frames']:.1%})")
    print(f"  Box center Δ median:   "
          f"({stability['center_dx_px_median']:.2f}, {stability['center_dy_px_median']:.2f}) px")
    print(f"  Box center Δ p95:      "
          f"({stability['center_dx_px_p95']:.2f}, {stability['center_dy_px_p95']:.2f}) px")
    print(f"  Box center Δ max:      "
          f"({stability['center_dx_px_max']:.2f}, {stability['center_dy_px_max']:.2f}) px")
    print(f"  Box size Δ median:     "
          f"({stability['size_dx_px_median']:.2f}, {stability['size_dy_px_median']:.2f}) px")

    # Pass 2: throughput at cadences.
    print(f"\n  Cadence  FPS     mean ms   p95 ms")
    print(f"  {'-'*34}")
    results = {}
    for K in (1, 2, 3, 4, 6):
        fps, mean_ms, p95_ms = bench_cadence(
            model, context, d_joints, d_verts, all_frames, K)
        results[K] = {"fps": round(fps, 2), "mean_ms": round(mean_ms, 3),
                      "p95_ms": round(p95_ms, 3)}
        gain = f"{fps/results[1]['fps']:.2f}x" if K > 1 else "1.00x"
        print(f"  {K:>7}  {fps:>7.1f} {mean_ms:>9.2f} {p95_ms:>9.2f}  {gain}")
    print("=" * 70)

    out = ROOT / "results" / "raw" / "box-cadence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "clips": CLIPS,
        "stability": stability,
        "cadence_fps": results,
    }, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()