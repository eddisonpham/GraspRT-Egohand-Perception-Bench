"""Fine-grained component profiler for the detector + crop bottleneck.

Uses wall-clock subtraction (no fragile ultralytics internals):
  yolo_call:     model.detector(img)                 (YOLO forward + bbox parse)
  preprocess:    model.preprocess(img)               (YOLO + ViTDetDataset arg setup)
  ds_build:      preprocess - yolo_call              (~ViTDetDataset construction)
  crop_index:    dataset[0]                          (actual crop/resize/normalize)
  crop_collate:  unsqueeze + recursive_to(cuda)      (CPU->GPU)
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
    N_IMAGES = int(os.environ.get("COMP_N_IMAGES", 30))
    WARMUP = 10

    import torch

    from data.freihand.loader import FreiHandLoader
    from models.wilor_wrapper import WiLoRHandModel

    model = WiLoRHandModel(variant="fast")
    model.load("cuda")

    loader = FreiHandLoader(subset="data/freihand/subsets/dev.json")
    images = [loader[i][0] for i in range(N_IMAGES)]

    for img in images[:WARMUP]:
        model.preprocess(img)

    t_yolo, t_prep, t_idx, t_coll = [], [], [], []
    missed = 0
    for img in images:
        t0 = time.perf_counter()
        _ = model.detector(img, conf=0.3, verbose=False)
        t_yolo.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        batch = model.preprocess(img)
        t_prep.append((time.perf_counter() - t0) * 1000)

        if batch["dataset"] is None:
            missed += 1
            t_idx.append(0)
            t_coll.append(0)
            continue

        t0 = time.perf_counter()
        item = batch["dataset"][0]
        item = {k: torch.as_tensor(v) if isinstance(v, np.ndarray) else v
                for k, v in item.items()}
        item = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.ndim >= 1
                and k in {"img", "right", "box_center", "box_size", "img_size"}
                else v for k, v in item.items()}
        from wilor.utils import recursive_to
        _ = recursive_to(item, model._torch_device)
        torch.cuda.synchronize()
        t_idx.append((time.perf_counter() - t0) * 1000)

    def summ(arr):
        a = np.asarray(arr) if arr else np.zeros(1)
        return {"mean_ms": round(float(a.mean()), 3),
                "p95_ms": round(float(np.percentile(a, 95)), 3)}

    ds_build = [p - y for p, y in zip(t_prep, t_yolo)]

    print("\n=== Detector + crop fine-grained profile (real FreiHAND) ===")
    labels = [
        ("yolo_call (forward+parse)", t_yolo),
        ("ds_build (ViTDetDataset args)", ds_build),
        ("crop_index (crop+normalize)", t_idx),
        ("preprocess total", t_prep),
    ]
    for name, arr in labels:
        s = summ(arr)
        print(f"  {name:<34} mean {s['mean_ms']:7.3f}  p95 {s['p95_ms']:7.3f}")
    print(f"  (misses: {missed}/{N_IMAGES})")

    out = ROOT / "results" / "raw" / "component-profile.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_images": N_IMAGES, "misses": missed,
        "substeps": {name: summ(arr) for name, arr in labels},
    }, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()