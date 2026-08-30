"""Export the WiLoR YOLO hand detector to TensorRT FP16 engine.

Uses ultralytics' built-in .export(format='engine') so input-letterbox,
pre/postprocess stay in the official code path. Then benchmarks the engine
vs the raw PyTorch detector on the same images.

The hypothesis: the 14ms detector is dominated by ultralytics Python/CPU
overhead, not GPU compute. A TRT engine removes the per-call PyTorch dispatch
but keeps the SAME preprocess/postprocess, isolating the GPU-side win.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    from data.freihand.loader import FreiHandLoader
    from models.wilor_wrapper import WiLoRHandModel

    model = WiLoRHandModel(variant="fast")
    model.load("cuda")
    det = model.detector
    loader = FreiHandLoader(subset="data/freihand/subsets/dev.json")
    images = [loader[i][0] for i in range(20)]

    det_path = str(model.root / "pretrained_models" / "detector.pt")
    engine_path = str(ROOT / "results" / "trt" / "yolo-hand-fp32.engine")
    half = os.environ.get("DET_HALF", "1") == "1"
    if not os.path.exists(engine_path):
        print(f"Exporting YOLO detector to TRT {'FP16' if half else 'FP32'} engine ...")
        # Import here so we load via ultralytics directly, not the wrapper.
        from ultralytics import YOLO
        det_model = YOLO(det_path)
        t0 = time.perf_counter()
        det_model.export(format="engine", half=half, imgsz=224,
                         device=0, simplify=True, workspace=3)
        exported = str(det_path).replace(".pt", ".engine")
        if os.path.exists(exported):
            import shutil
            os.makedirs(Path(engine_path).parent, exist_ok=True)
            shutil.copy(exported, engine_path)  # cross-device safe
        print(f"export took {time.perf_counter() - t0:.1f}s -> {engine_path}")

    # --- Benchmark PyTorch detector (baseline) ---
    torch_times = []
    for img in images[:5]:
        det(img, verbose=False)
    for img in images:
        t0 = time.perf_counter()
        det(img, verbose=False)
        torch_times.append((time.perf_counter() - t0) * 1000)

    # --- Benchmark TRT engine (via ultralytics YOLO loading the engine) ---
    from ultralytics import YOLO
    trt_det = YOLO(engine_path, task="detect")
    trt_times = []
    for img in images[:5]:
        trt_det(img, verbose=False)
    for img in images:
        t0 = time.perf_counter()
        trt_det(img, verbose=False)
        trt_times.append((time.perf_counter() - t0) * 1000)

    # --- Accuracy proxy: detection rate + max conf must match ---
    def stats(det_model):
        counts, confs = [], []
        for img in images:
            r = det_model(img, verbose=False)[0]
            n = 0 if r.boxes is None else len(r.boxes)
            counts.append(1 if n > 0 else 0)
            if n:
                confs.append(float(r.boxes.conf.max()))
        return float(np.mean(counts)), float(np.mean(confs)) if confs else 0.0

    torch_det, torch_conf = stats(det)
    trt_det_r, trt_conf = stats(trt_det)

    def summ(arr):
        a = np.asarray(arr)
        return {"mean_ms": round(float(a.mean()), 3),
                "p95_ms": round(float(np.percentile(a, 95)), 3)}

    print("\n=== YOLO detector: PyTorch vs TRT FP16 engine ===")
    for name, arr in [("PyTorch detector", torch_times),
                      ("TRT FP16 detector", trt_times)]:
        s = summ(arr)
        print(f"  {name:<20} mean {s['mean_ms']:7.3f}  p95 {s['p95_ms']:7.3f}")
    print(f"  detection rate  PyTorch={torch_det:.3f} conf={torch_conf:.3f} | "
          f"TRT={trt_det_r:.3f} conf={trt_conf:.3f}")

    out = ROOT / "results" / "raw" / "detector-trt.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "engine": engine_path,
        "pytorch": {"latency": summ(torch_times),
                    "detection_rate": torch_det, "conf": torch_conf},
        "tensorrt": {"latency": summ(trt_times),
                     "detection_rate": trt_det_r, "conf": trt_conf},
    }, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()