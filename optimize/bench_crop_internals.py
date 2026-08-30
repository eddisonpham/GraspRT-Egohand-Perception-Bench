"""Profile ViTDetDataset __getitem__ internals to find CPU hotspots.

Breaks the ~6.8ms crop stage into: skimage gaussian blur, cv2 image-patch
warp, tensor conversion, normalization. Pure wall-clock instrumentation that
mirrors the exact __getitem__ body, so any speedup found is accuracy-preserving.
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
    import cv2

    from data.freihand.loader import FreiHandLoader
    from models.wilor_wrapper import WiLoRHandModel

    N = 30
    model = WiLoRHandModel(variant="fast")
    model.load("cuda")

    loader = FreiHandLoader(subset="data/freihand/subsets/dev.json")
    images = [loader[i][0] for i in range(N)]

    # Namespace mirroring ViTDetDataset preprocess (from wilor cfg)
    cfg = model.cfg
    img_size = cfg.MODEL.IMAGE_SIZE
    mean = 255.0 * np.asarray(cfg.MODEL.IMAGE_MEAN)
    std = 255.0 * np.asarray(cfg.MODEL.IMAGE_STD)
    BBOX_SHAPE = cfg.MODEL.get("BBOX_SHAPE", None)

    from wilor.datasets.utils import convert_cvimg_to_tensor, expand_to_aspect_ratio, generate_image_patch_cv2
    from skimage.filters import gaussian

    t_blur, t_warp, t_convert, t_norm, t_total = [], [], [], [], []
    for img in images[:5]:
        # warm: run a full preprocess to make a dataset
        model.preprocess(img)
    for img in images:
        t0 = time.perf_counter()
        raw = img
        # mock a box (center ~112) to force the crop path
        center = np.array([112.0, 112.0])
        scale = np.array(2.0)
        bbox_size = expand_to_aspect_ratio(scale * 200, target_aspect_ratio=BBOX_SHAPE).max()
        patch_width = patch_height = img_size
        right = 1.0

        t0b = time.perf_counter()
        downsampling_factor = (bbox_size * 1.0) / patch_width
        downsampling_factor = downsampling_factor / 2.0
        if downsampling_factor > 1.1:
            blurbed = gaussian(raw, sigma=(downsampling_factor - 1) / 2,
                               channel_axis=2, preserve_range=True)
        else:
            blurbed = raw
        t1 = time.perf_counter()
        t_blur.append((t1 - t0b) * 1000)

        img_patch, _ = generate_image_patch_cv2(
            blurbed, 112.0, 112.0, bbox_size, bbox_size,
            patch_width, patch_height, False, 1.0, 0,
            border_mode=cv2.BORDER_CONSTANT)
        img_patch = img_patch[:, :, ::-1]
        t2 = time.perf_counter()
        t_warp.append((t2 - t1) * 1000)

        img_t = convert_cvimg_to_tensor(img_patch)
        t3 = time.perf_counter()
        t_convert.append((t3 - t2) * 1000)

        for n_c in range(min(raw.shape[2], 3)):
            img_t[n_c, :, :] = (img_t[n_c, :, :] - mean[n_c]) / std[n_c]
        t4 = time.perf_counter()
        t_norm.append((t4 - t3) * 1000)
        t_total.append((t4 - t0) * 1000)

    def summ(arr):
        a = np.asarray(arr) if arr else np.zeros(1)
        return {"mean_ms": round(float(a.mean()), 3),
                "p95_ms": round(float(np.percentile(a, 95)), 3)}

    print("\n=== ViTDetDataset crop internals (30 imgs) ===")
    for name, arr in [("skimage blur", t_blur), ("cv2 warp+flip", t_warp),
                      ("tensor convert", t_convert), ("normalize", t_norm),
                      ("crop total", t_total)]:
        s = summ(arr)
        print(f"  {name:<18} mean {s['mean_ms']:7.3f}  p95 {s['p95_ms']:7.3f}")

    out = ROOT / "results" / "raw" / "crop-internals.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_images": N,
        "substeps": {name: summ(arr) for name, arr in
                     [("blur", t_blur), ("warp", t_warp), ("convert", t_convert),
                      ("normalize", t_norm), ("total", t_total)]},
    }, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()