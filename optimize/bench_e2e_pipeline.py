"""End-to-end pipeline profiling: YOLO detector + crop + TRT FP16 reconstruction.

Measures the TRUE wall-clock production FPS of the full pipeline on real
FreiHAND images, with per-stage breakdown and ResourceMonitor profiling.

Stages:
  1. Detector: YOLO hand detection on raw 224x224 image
  2. Crop: ViTDetDataset normalization (CPU → GPU tensor)
  3. TRT reconstruction: MANO joints + mesh from the TRT FP16 engine
  4. Postprocess: detach + CPU transfer

Reports: total wall-clock FPS, per-stage latency, GPU util, power draw.

Usage (WSL, egohand env):
  WILOR_ROOT=~/src/WiLoR FREIHAND_ROOT=~/egohand_data/freihand \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python optimize/bench_e2e_pipeline.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WARMUP = 10
N_IMAGES = 100


def main() -> None:
    import tensorrt as trt
    import torch

    from common.profiling import ResourceMonitor, cpu_load_sample, nvidia_smi_snapshot
    from data.freihand.loader import FreiHandLoader
    from models.wilor_wrapper import WiLoRHandModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="fast")
    ap.add_argument("--n-images", type=int, default=N_IMAGES)
    ap.add_argument("--warmup", type=int, default=WARMUP)
    args = ap.parse_args()

    # --- Load model + TRT engine ---
    model = WiLoRHandModel(variant=args.variant)
    model.load("cuda")
    eager = getattr(model.model.backbone, "_orig_mod", None)
    if eager is not None:
        model.model.backbone = eager
    model.model.requires_grad_(False)

    engine_path = str(ROOT / "results" / "trt" / "wilor-fast-fp16.plan")
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    d_joints = torch.empty((1, 21, 3), dtype=torch.float16, device="cuda")
    d_verts = torch.empty((1, 778, 3), dtype=torch.float16, device="cuda")

    # --- Load images ---
    loader = FreiHandLoader(subset="data/freihand/subsets/dev.json")
    n = min(args.n_images, len(loader))
    images = [loader[i][0] for i in range(n)]

    # --- Per-stage timing arrays ---
    times_det = []      # detector (YOLO)
    times_crop = []     # ViTDetDataset crop/normalize
    times_trt = []      # TRT inference
    times_post = []     # CPU transfer + detach
    times_total = []    # full pipeline wall-clock

    # --- Warmup ---
    for img in images[:args.warmup]:
        batch = model.preprocess(img)
        if batch["dataset"] is not None:
            item = batch["dataset"][0]
            item = {k: torch.as_tensor(v) if isinstance(v, np.ndarray) else v
                    for k, v in item.items()}
            item = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.ndim >= 1
                    and k in {"img", "right", "box_center", "box_size", "img_size"}
                    else v for k, v in item.items()}
            from wilor.utils import recursive_to
            item = recursive_to(item, model._torch_device)
            img_t = item["img"]
            h_input = img_t.to(torch.float16)
            context.execute_v2(bindings=[
                h_input.data_ptr(), d_joints.data_ptr(), d_verts.data_ptr()])
            torch.cuda.synchronize()

    # --- Resource monitor ---
    mon = ResourceMonitor(interval_s=0.025)
    mon.start()
    torch.cuda.reset_peak_memory_stats()
    stream = torch.cuda.Stream()

    for img in images[args.warmup:]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Stage 1: detector + preprocessing (includes YOLO + ViTDetDataset)
        t_s = time.perf_counter()
        batch = model.preprocess(img)
        torch.cuda.synchronize()
        t_det = time.perf_counter()
        times_det.append((t_det - t_s) * 1000)

        if batch["dataset"] is None:
            # detector miss — record full pipeline time and skip reconstruction
            times_crop.append(0)
            times_trt.append(0)
            times_post.append(0)
            times_total.append((time.perf_counter() - t0) * 1000)
            continue

        # Stage 2: crop/collate to GPU tensor
        t_s = time.perf_counter()
        item = batch["dataset"][0]
        item = {k: torch.as_tensor(v) if isinstance(v, np.ndarray) else v
                for k, v in item.items()}
        item = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.ndim >= 1
                and k in {"img", "right", "box_center", "box_size", "img_size"}
                else v for k, v in item.items()}
        from wilor.utils import recursive_to
        item = recursive_to(item, model._torch_device)
        torch.cuda.synchronize()
        t_crop_end = time.perf_counter()
        times_crop.append((t_crop_end - t_s) * 1000)

        # Stage 3: TRT reconstruction
        t_s = time.perf_counter()
        img_t = item["img"].to(torch.float16)
        with torch.cuda.stream(stream):
            context.execute_v2(bindings=[
                img_t.data_ptr(), d_joints.data_ptr(), d_verts.data_ptr()])
        torch.cuda.synchronize()
        t_trt_end = time.perf_counter()
        times_trt.append((t_trt_end - t_s) * 1000)

        # Stage 4: postprocess (CPU transfer)
        t_s = time.perf_counter()
        _joints = d_joints.cpu().numpy()
        _verts = d_verts.cpu().numpy()
        torch.cuda.synchronize()
        times_post.append((time.perf_counter() - t_s) * 1000)

        times_total.append((time.perf_counter() - t0) * 1000)

    mon.stop()

    # --- Aggregate ---
    def stats(arr):
        if not arr:
            return {"mean_ms": 0, "p95_ms": 0, "std_ms": 0}
        a = np.asarray(arr)
        return {"mean_ms": round(float(a.mean()), 3),
                "p95_ms": round(float(np.percentile(a, 95)), 3),
                "std_ms": round(float(a.std()), 3)}

    n_total = len(times_total)
    n_detected = sum(1 for d, t in zip(times_det, times_total) if t > 0)

    resource = mon.summary()
    vram_peak = round(torch.cuda.max_memory_allocated() / 1e6, 1)

    total_mean = np.mean(times_total) if times_total else 999
    e2e_fps = round(1000 / total_mean, 1)

    print("\n" + "=" * 70)
    print("END-TO-END PIPELINE PROFILING")
    print("=" * 70)
    print(f"  Images processed:    {n_total}")
    print(f"  Detected:            {n_detected}/{n_total} ({n_detected/n_total:.1%})")
    print(f"  Wall-clock mean:     {total_mean:.1f} ms")
    print(f"  E2E FPS:             {e2e_fps}")
    print()
    print(f"  {'Stage':<25} {'Mean ms':>10} {'p95 ms':>10} {'% of total':>10}")
    print(f"  {'-'*55}")
    for label, arr in [("Detector+preprocess", times_det),
                       ("Crop/collate", times_crop),
                       ("TRT reconstruction", times_trt),
                       ("Postprocess (CPU)", times_post)]:
        s = stats(arr)
        pct = s["mean_ms"] / total_mean * 100 if total_mean > 0 else 0
        print(f"  {label:<25} {s['mean_ms']:>10.1f} {s['p95_ms']:>10.1f} {pct:>9.1f}%")
    print(f"  {'TOTAL':<25} {total_mean:>10.1f} "
          f"{stats(times_total)['p95_ms']:>10.1f} {'100.0':>9}%")
    print()
    print(f"  GPU util mean:       {resource.get('gpu_util_pct_mean', 'n/a')}")
    print(f"  GPU util peak:       {resource.get('gpu_util_pct_peak', 'n/a')}")
    print(f"  Power mean:          {resource.get('power_watts_mean', 'n/a')}W")
    print(f"  Temp peak:           {resource.get('temp_c_peak', 'n/a')}C")
    print(f"  VRAM peak:           {vram_peak} MB")
    print(f"  CPU load 1m:         {resource.get('load_1m_mean', 'n/a')}")
    print("=" * 70)

    out = ROOT / "results" / "raw" / "e2e-pipeline-profiled.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "variant": args.variant,
        "engine": engine_path,
        "n_images": n_total,
        "n_detected": n_detected,
        "e2e_fps": e2e_fps,
        "stages": {
            "detector_preprocess": stats(times_det),
            "crop_collate": stats(times_crop),
            "trt_reconstruction": stats(times_trt),
            "postprocess": stats(times_post),
            "total": stats(times_total),
        },
        "resource": resource,
        "vram_peak_mb": vram_peak,
    }, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
