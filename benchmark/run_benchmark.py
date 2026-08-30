"""Unified benchmark entrypoint.

Run one candidate per process to prevent allocator fragmentation and enforce the 7GB
constraint. Model modules are imported lazily because MediaPipe is native Windows while
GPU candidates run in WSL-specific environments.

Examples:
  python benchmark/run_benchmark.py --model mediapipe --subset dev --iters 200
  python benchmark/run_benchmark.py --model mediapipe --subset full --iters 200
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_IMPORTS = {
    "mediapipe": ("models.mediapipe_wrapper", "MediaPipeHandModel"),
    "mobrecon": ("models.mobrecon_wrapper", "MobReconHandModel"),
    "wilor": ("models.wilor_wrapper", "WiLoRHandModel"),
    "hamer": ("models.hamer_wrapper", "HamerHandModel"),
}


def model_size_mb(model_name: str, extra_roots: list[Path] | None = None) -> float | None:
    """Sum checkpoint files under conventional local/external model directories."""
    candidates = [ROOT / "models" / "weights", ROOT / "pretrained_models", ROOT / "checkpoints"]
    if model_name == "mobrecon":
        candidates.append(Path(os.environ.get("MOBRECON_ROOT", str(Path.home() / "src" / "HandMesh"))) / "downloads")
    if extra_roots:
        candidates.extend(extra_roots)
    total = 0
    for base in candidates:
        if base.exists():
            for p in base.rglob("*"):
                if p.is_file() and p.suffix.lower() in {".pt", ".pth", ".ckpt", ".onnx", ".bin"}:
                    total += p.stat().st_size
    return total / 1e6 if total else None


def run(args):
    mod_name, cls_name = MODEL_IMPORTS[args.model]
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    model = cls(variant=args.variant) if args.variant else cls()
    model.load(device=args.device)

    from data.freihand.loader import FreiHandLoader
    from common.metrics import pa_mpjpe, pa_mpvpe, f_score
    from common.profiling import NvidiaSmiMonitor, time_infer, torch_peak_mb

    subset_path = ROOT / "data" / "freihand" / "subsets" / f"{args.subset}.json"
    # Prefer WSL ext4 data root; loader handles this when FREIHAND_ROOT is unset.
    loader = FreiHandLoader(root=args.root, subset=None) if args.root else FreiHandLoader(subset=subset_path)
    # For timing, choose first image with a prediction if possible; for accuracy, all images.
    sample_img, _, _ = loader[0]
    sample_batch = model.preprocess(sample_img)
    model.warmup(n=args.warmup)

    # Separate process + reset after model load: allocator peak measures inference workload,
    # while nvidia-smi captures actual process/model footprint.
    smi = NvidiaSmiMonitor(interval_s=0.05) if args.device != "cpu" else None
    if args.device != "cpu":
        import torch
        torch.cuda.reset_peak_memory_stats()
        smi.start()
    latency = time_infer(model, sample_batch, n_warmup=0, n_iters=args.iters,
                         use_cuda_events=args.device != "cpu")
    if smi:
        smi.stop()
        vram = {"torch_peak": torch_peak_mb(), "nvidia_smi_peak": smi.peak_mb,
                "nvidia_smi_samples": smi.samples_mb}
    else:
        vram = {"torch_peak": None, "nvidia_smi_peak": None, "nvidia_smi_samples": []}

    # Accuracy: first prediction only (single dominant hand protocol); count misses.
    errors, vert_errors, f5, f15 = [], [], [], []
    misses = 0
    for i in range(len(loader)):
        image, gt_joints, _ = loader[i]
        preds = model.infer(model.preprocess(image))
        if not preds:
            misses += 1
            continue
        p = preds[0]
        errors.append(pa_mpjpe(p.joints_3d, gt_joints))
        if p.mesh_verts is not None:
            gt_v = loader.get_gt_verts(i)
            vert_errors.append(pa_mpvpe(p.mesh_verts, gt_v))
            f5.append(f_score(p.mesh_verts, gt_v, 5.0))
            f15.append(f_score(p.mesh_verts, gt_v, 15.0))

    def mean_std(xs):
        return {"mean": float(np.mean(xs)) if xs else None,
                "std": float(np.std(xs)) if xs else None}

    payload = {
        "model": args.model,
        "variant": args.variant or "default",
        "commit": os.environ.get("MODEL_COMMIT", "unknown"),
        "protocol": {"warmup": args.warmup, "timed_iterations": args.iters,
                      "single_dominant_hand": True, "vram_ceiling_mb": 6000},
        "device": f"{args.device} ({platform.node()})",
        "latency_ms": {"mean": latency["mean_ms"], "median": latency["median_ms"],
                       "p95": latency["p95_ms"], "std": latency["std_ms"]},
        "fps": 1000.0 / latency["mean_ms"] if latency["mean_ms"] > 0 else 0.0,
        "vram_mb": vram,
        "accuracy": {
            "subset": args.subset,
            "pa_mpjpe_mm": mean_std(errors)["mean"],
            "pa_mpjpe_std_mm": mean_std(errors)["std"],
            "pa_mpvpe_mm": mean_std(vert_errors)["mean"] if vert_errors else None,
            "pa_mpvpe_std_mm": mean_std(vert_errors)["std"] if vert_errors else None,
            "f_score_5mm": mean_std(f5)["mean"] if f5 else None,
            "f_score_15mm": mean_std(f15)["mean"] if f15 else None,
        },
        "model_size_mb": model_size_mb(args.model, [Path(os.environ["WILOR_ROOT"]) / "pretrained_models"] if os.environ.get("WILOR_ROOT") else None),
        "n_eval_images": len(loader),
        "n_predictions": len(errors),
        "n_misses": misses,
        "notes": args.notes,
    }
    out = ROOT / "results" / "raw" / f"{args.model}{('-' + args.variant) if args.variant else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODEL_IMPORTS, required=True)
    p.add_argument("--variant", default=None)
    p.add_argument("--subset", choices=["dev", "full"], default="dev")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--notes", default="")
    p.add_argument("--root", default=None, help="Explicit FreiHAND root (useful for native Windows sparse copy)")
    run(p.parse_args())


if __name__ == "__main__":
    main()