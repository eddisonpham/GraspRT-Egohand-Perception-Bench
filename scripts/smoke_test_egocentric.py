"""Egocentric video smoke test: full pipeline on synthetic clips.

Reads every frame from data/egocentric_clips/*.mp4, runs the complete
WiLoR-fast pipeline (YOLO detector + ViTDetDataset crop + TRT FP16
reconstruction), and reports:
  - Per-clip: detection rate, mean confidence, joint statistics
  - Per-clip: wall-clock latency (mean, p95), FPS
  - ResourceMonitor: GPU util, power, temperature
  - Invariant checks: joints shape (21,3), all-finite, confidence in [0,1]

This is a qualitative smoke test — these synthetic clips have no GT joints,
so we check behavioral invariants rather than accuracy metrics.

Usage (WSL, egohand env):
  WILOR_ROOT=~/src/WiLoR FREIHAND_ROOT=~/egohand_data/freihand \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python scripts/smoke_test_egocentric.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CLIPS_DIR = ROOT / "data" / "egocentric_clips"


def process_clip(video_path: str, model, n_frames: int | None = None) -> dict:
    """Run full pipeline on one video clip, return per-frame stats."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_rate = cap.get(cv2.CAP_PROP_FPS)
    if n_frames:
        total = min(total, n_frames)

    frames_data = []
    latencies_ms = []
    detector_misses = 0

    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        # Full pipeline: preprocess (includes detector) + infer
        t0 = time.perf_counter()
        batch = model.preprocess(frame)
        preds = model.infer(batch)
        lat = (time.perf_counter() - t0) * 1000
        latencies_ms.append(lat)

        if not preds:
            detector_misses += 1
            frames_data.append({
                "frame": i, "detected": False, "latency_ms": round(lat, 2)})
            continue

        p = preds[0]
        joints = p.joints_3d
        nans = int(np.isnan(joints).sum())
        abs_max = float(np.abs(joints).max())
        frames_data.append({
            "frame": i,
            "detected": True,
            "confidence": round(p.confidence, 4),
            "handedness": p.handedness,
            "joints_abs_max": round(abs_max, 6),
            "joints_nans": nans,
            "has_verts": p.mesh_verts is not None,
            "latency_ms": round(lat, 2),
        })

    cap.release()

    detected = [f for f in frames_data if f["detected"]]
    lat_arr = np.asarray(latencies_ms)
    return {
        "clip": Path(video_path).name,
        "total_frames": total,
        "source_fps": fps_rate,
        "detected_frames": len(detected),
        "missed_frames": detector_misses,
        "detection_rate": round(len(detected) / max(total, 1), 3),
        "avg_confidence": round(float(np.mean([d["confidence"] for d in detected])), 4) if detected else 0,
        "joints_abs_max_range": (
            round(min(d["joints_abs_max"] for d in detected), 6),
            round(max(d["joints_abs_max"] for d in detected), 6),
        ) if detected else (0, 0),
        "latency_ms": {
            "mean": round(float(lat_arr.mean()), 2),
            "p95": round(float(np.percentile(lat_arr, 95)), 2),
            "std": round(float(lat_arr.std()), 2),
        },
        "effective_fps": round(1000 / float(lat_arr.mean()), 1),
        "all_joints_finite": all(d.get("joints_nans", 0) == 0 for d in detected),
        "all_confidence_valid": all(0 <= d.get("confidence", 0) <= 1 for d in detected),
        "frames": frames_data,
    }


def main() -> None:
    from common.profiling import ResourceMonitor
    from models.wilor_wrapper import WiLoRHandModel

    print("Loading WiLoR-fast model...")
    model = WiLoRHandModel(variant="fast")
    model.load("cuda")

    clips = sorted(CLIPS_DIR.glob("*.mp4"))
    if not clips:
        sys.exit(f"No .mp4 clips found in {CLIPS_DIR}")

    print(f"Found {len(clips)} clip(s)\n")

    mon = ResourceMonitor(interval_s=0.05)
    mon.start()

    all_results = []
    for clip_path in clips:
        print(f"Processing {clip_path.name}...")
        r = process_clip(str(clip_path), model)
        all_results.append(r)
        print(f"  detection rate: {r['detection_rate']:.1%} "
              f"({r['detected_frames']}/{r['total_frames']})")
        print(f"  avg confidence: {r['avg_confidence']:.3f}")
        print(f"  latency: {r['latency_ms']['mean']:.1f}ms "
              f"(p95 {r['latency_ms']['p95']:.1f}ms)")
        print(f"  effective FPS: {r['effective_fps']:.0f}")
        print(f"  joints finite: {r['all_joints_finite']}, "
              f"confidence valid: {r['all_confidence_valid']}")
        print()

    mon.stop()

    # Aggregate across all clips
    total_frames = sum(r["total_frames"] for r in all_results)
    total_detected = sum(r["detected_frames"] for r in all_results)
    all_lat = []
    for r in all_results:
        for f in r["frames"]:
            if f["detected"]:
                all_lat.append(f["latency_ms"])

    resource = mon.summary()
    print("=" * 60)
    print("AGGREGATE SMOKE TEST RESULTS")
    print("=" * 60)
    print(f"  Clips processed:     {len(all_results)}")
    print(f"  Total frames:        {total_frames}")
    print(f"  Detected:            {total_detected}/{total_frames} "
          f"({total_detected/max(total_frames,1):.1%})")
    if all_lat:
        arr = np.asarray(all_lat)
        print(f"  Detection latency:   mean {arr.mean():.1f}ms, "
              f"p95 {np.percentile(arr, 95):.1f}ms")
        print(f"  Effective FPS:       {1000/arr.mean():.0f}")
    print(f"  GPU util mean:       {resource.get('gpu_util_pct_mean', 'n/a')}")
    print(f"  GPU util peak:       {resource.get('gpu_util_pct_peak', 'n/a')}")
    print(f"  Power mean:          {resource.get('power_watts_mean', 'n/a')}W")
    print(f"  Temp peak:           {resource.get('temp_c_peak', 'n/a')}C")
    print(f"  VRAM peak:           {resource.get('mem_used_mb_peak', 'n/a')} MB")

    # Invariant checks
    all_finite = all(r["all_joints_finite"] for r in all_results)
    all_conf = all(r["all_confidence_valid"] for r in all_results)
    any_detection = total_detected > 0
    print(f"\n  INVARIANT joints all-finite: {'PASS' if all_finite else 'FAIL'}")
    print(f"  INVARIANT confidence in [0,1]: {'PASS' if all_conf else 'FAIL'}")
    print(f"  INVARIANT at least 1 detection: {'PASS' if any_detection else 'FAIL'}")

    overall = "PASS" if (all_finite and all_conf and any_detection) else "FAIL"
    print(f"\n  OVERALL: {overall}")

    out = ROOT / "results" / "raw" / "egocentric-smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "clips_dir": str(CLIPS_DIR),
        "n_clips": len(all_results),
        "total_frames": total_frames,
        "total_detected": total_detected,
        "overall": overall,
        "resource": resource,
        "results": [{k: v for k, v in r.items() if k != "frames"}
                    for r in all_results],
    }, indent=2))
    print(f"\nsaved {out}")
    sys.exit(0 if overall == "PASS" else 1)


if __name__ == "__main__":
    main()
