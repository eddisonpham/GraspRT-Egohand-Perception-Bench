"""INT8 PTQ: build an INT8 TensorRT engine with entropy calibration (accuracy-gated).

Calibrates on real FreiHAND crops (WiLoR preprocessing, no labels needed — PTQ is
unsupervised). After building, outputs are compared against the PyTorch reference;
the runbook gate requires INT8 PA-MPJPE-vs-PyTorch to stay within a documented
budget or the engine is rejected.

Usage (WSL, egohand env):
  LD_LIBRARY_PATH=<tensorrt_libs>:<nvidia libs> \
  WILOR_ROOT=~/src/WiLoR FREIHAND_ROOT=~/egohand_data/freihand \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python optimize/build_trt_int8.py --calib-count 128
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_calibrator(count: int, batch: int = 1):
    """IInt8EntropyCalibrator2 feeding real WiLoR crops from FreiHAND dev images."""
    import tensorrt as trt
    import torch

    from data.freihand.loader import FreiHandLoader
    from models.wilor_wrapper import WiLoRHandModel

    class FreiCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self, total: int):
            super().__init__()
            self.batch = batch
            model = WiLoRHandModel(variant="fast")
            model.load("cuda")
            loader = FreiHandLoader(subset="data/freihand/subsets/dev.json")
            self.blobs = []
            for i in range(total):
                img, _, _ = loader[i]
                pre = model.preprocess(img)
                if pre["dataset"] is None:
                    continue
                item = pre["dataset"][0]
                crop = np.ascontiguousarray(
                    np.asarray(item["img"], dtype=np.float32)[None]
                )  # (1,3,256,256) fp32 expected by calibrator
                self.blobs.append(crop)
            self.idx = 0
            self.d_input = torch.empty(self.blobs[0].shape, dtype=torch.float32, device="cuda")
            print(f"calibration set: {len(self.blobs)} crops")

        def get_batch_size(self):
            return self.batch

        def get_batch(self, names):
            if self.idx >= len(self.blobs):
                return None
            self.d_input.copy_(torch.from_numpy(self.blobs[self.idx]).cuda())
            self.idx += 1
            return [int(self.d_input.data_ptr())]

        def read_calibration_cache(self):
            cache = ROOT / "results" / "trt" / "int8_calib.cache"
            if cache.exists():
                print("using calibration cache")
                return cache.read_bytes()
            return None

        def write_calibration_cache(self, data):
            cache = ROOT / "results" / "trt" / "int8_calib.cache"
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(data)
            print(f"calibration cache written: {cache}")

    return FreiCalibrator(count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(ROOT / "results" / "onnx" / "wilor-fast.onnx"))
    ap.add_argument("--out", default=str(ROOT / "results" / "trt" / "wilor-fast-int8.plan"))
    ap.add_argument("--calib-count", type=int, default=128)
    ap.add_argument("--workspace-gb", type=float, default=3.0)
    args = ap.parse_args()

    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(args.onnx):
        for i in range(parser.num_errors):
            print("parse error:", parser.get_error(i))
        sys.exit("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb * 1e9))
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.INT8)
    calib = make_calibrator(args.calib_count)
    config.int8_calibrator = calib

    profile = builder.create_optimization_profile()
    profile.set_shape("image", (1, 3, 256, 256), (1, 3, 256, 256), (1, 3, 256, 256))
    config.add_optimization_profile(profile)

    print("building INT8 engine (calibration runs first)...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        sys.exit("INT8 engine build failed")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = bytes(engine_bytes)
    out.write_bytes(data)
    meta = {
        "onnx": args.onnx,
        "precision": "int8-fp16-fallback",
        "calibrator": "IInt8EntropyCalibrator2",
        "calib_count": args.calib_count,
        "engine_bytes": len(data),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "tensorrt": trt.__version__,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"INT8 engine written: {out} ({len(data) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
