"""End-to-end: torch detector vs TRT detector, both with TRT reconstruction.

Measures the real FPS gain of the TRT detector AND verifies PA-MPJPE accuracy
is preserved, on real FreiHAND dev images. This is the decisive test for
adopting the TRT detector.
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

N_IMAGES = 100
WARMUP = 10


class TRTDetectorPipeline:
    """Reconstruct using a TRT YOLO detector + TRT reconstruction engine."""

    def __init__(self, model, detector_engine: str, recon_engine: str):
        from ultralytics import YOLO
        self.model = model
        self.detector = YOLO(detector_engine, task="pose")
        self.recon_engine = recon_engine

    def preprocess(self, img):
        # Use the same ViTDetDataset path as the wrapper, but TRT detector.
        from wilor.datasets.vitdet_dataset import ViTDetDataset
        import numpy as np
        detections = self.detector(img, conf=0.3, verbose=False)[0]
        boxes, right = [], []
        if detections.boxes is not None:
            data = detections.boxes.data
            data = data.detach().cpu().squeeze().numpy()
            data = np.atleast_2d(data)
            cls = detections.boxes.cls.detach().cpu().numpy()
            for row, c in zip(data, np.atleast_1d(cls)):
                boxes.append(row[:4].tolist())
                right.append(float(c))
        if not boxes:
            return {"dataset": None, "image": img}
        ds = ViTDetDataset(self.model.cfg, img, np.asarray(boxes, dtype=np.float32),
                           np.asarray(right, dtype=np.float32), rescale_factor=2.0,
                           fp16=self.model.variant == "fast")
        return {"dataset": ds, "image": img}

    def infer_trt(self, batch):
        if batch["dataset"] is None:
            return None
        import torch
        import tensorrt as trt
        from wilor.utils import recursive_to
        # Build engine lazily on first use.
        if not hasattr(self, "_ctx"):
            logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger)
            with open(self.recon_engine, "rb") as f:
                eng = runtime.deserialize_cuda_engine(f.read())
            self._ctx = eng.create_execution_context()
            self._d_joints = torch.empty((1, 21, 3), dtype=torch.float16, device="cuda")
            self._d_verts = torch.empty((1, 778, 3), dtype=torch.float16, device="cuda")
        item = batch["dataset"][0]
        item = {k: torch.as_tensor(v) if isinstance(v, np.ndarray) else v
                for k, v in item.items()}
        item = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.ndim >= 1
                and k in {"img", "right", "box_center", "box_size", "img_size"}
                else v for k, v in item.items()}
        item = recursive_to(item, self.model._torch_device)
        img_t = item["img"].to(torch.float16)
        self._ctx.execute_v2(bindings=[img_t.data_ptr(),
                                       self._d_joints.data_ptr(),
                                       self._d_verts.data_ptr()])
        torch.cuda.synchronize()
        joints = self._d_joints.float().cpu().numpy().reshape(21, 3)
        verts = self._d_verts.float().cpu().numpy().reshape(778, 3)
        return joints, verts


def main() -> None:
    import torch
    from common.metrics import pa_mpjpe
    from data.freihand.loader import FreiHandLoader
    from models.wilor_wrapper import WiLoRHandModel

    engine_path = str(ROOT / "results" / "trt" / "wilor-fast-fp16.plan")
    det_engine = str(ROOT / "results" / "trt" / "yolo-hand-fp16.engine")

    model = WiLoRHandModel(variant="fast")
    model.load("cuda")
    # Load the reconstruction TRT engine into the same WiLoR model object.
    # We reuse model.preprocess for the torch path; for TRT we swap the detector.
    eager = getattr(model.model.backbone, "_orig_mod", None)
    if eager is not None:
        model.model.backbone = eager
    model.model.requires_grad_(False)

    trt_pipe = TRTDetectorPipeline(model, det_engine, engine_path)

    loader = FreiHandLoader(subset="data/freihand/subsets/dev.json")
    images = []
    gts = []
    for i in range(min(N_IMAGES, len(loader))):
        img, gt, _ = loader[i]
        images.append(img)
        gts.append(gt)

    # --- Torch detector path (model.preprocess has torch YOLO) ---
    for img in images[:WARMUP]:
        model.infer(model.preprocess(img))
    torch_det_errs, torch_det_times = [], []
    for img, gt in zip(images, gts):
        t0 = time.perf_counter()
        preds = model.infer(model.preprocess(img))
        torch_det_times.append((time.perf_counter() - t0) * 1000)
        if preds:
            torch_det_errs.append(pa_mpjpe(preds[0].joints_3d, gt))

    # --- TRT detector + TRT recon path ---
    trt_pipe.detector(images[0], verbose=False)  # init
    for img in images[:WARMUP]:
        trt_pipe.infer_trt(trt_pipe.preprocess(img))
    trt_det_errs, trt_det_times = [], []
    misses = 0
    for img, gt in zip(images, gts):
        t0 = time.perf_counter()
        out = trt_pipe.infer_trt(trt_pipe.preprocess(img))
        trt_det_times.append((time.perf_counter() - t0) * 1000)
        if out is None:
            misses += 1
            continue
        joints, _ = out
        trt_det_errs.append(pa_mpjpe(joints, gt))

    def summ(arr):
        a = np.asarray(arr)
        return {"mean_ms": round(float(a.mean()), 3),
                "p95_ms": round(float(np.percentile(a, 95)), 3)}

    def err(arr):
        a = np.asarray(arr)
        return round(float(a.mean()), 3)

    print("\n=== End-to-end: torch detector vs TRT detector (both TRT recon) ===")
    print(f"  torch detector: mean {summ(torch_det_times)['mean_ms']:.1f}ms "
          f"p95 {summ(torch_det_times)['p95_ms']:.1f}ms  "
          f"PA-MPJPE {err(torch_det_errs):.3f}mm  "
          f"FPS {1000/summ(torch_det_times)['mean_ms']:.1f}")
    print(f"  TRT detector:   mean {summ(trt_det_times)['mean_ms']:.1f}ms "
          f"p95 {summ(trt_det_times)['p95_ms']:.1f}ms  "
          f"PA-MPJPE {err(trt_det_errs):.3f}mm  "
          f"FPS {1000/summ(trt_det_times)['mean_ms']:.1f}  "
          f"misses {misses}")

    out = ROOT / "results" / "raw" / "e2e-trt-detector.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_images": N_IMAGES,
        "torch_detector": {"latency": summ(torch_det_times),
                           "pa_mpjpe_mm": err(torch_det_errs),
                           "fps": round(1000 / summ(torch_det_times)["mean_ms"], 1)},
        "trt_detector": {"latency": summ(trt_det_times),
                         "pa_mpjpe_mm": err(trt_det_errs),
                         "fps": round(1000 / summ(trt_det_times)["mean_ms"], 1),
                         "misses": misses},
    }, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()