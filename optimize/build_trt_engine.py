"""Build a native TensorRT FP16 engine from the exported WiLoR ONNX graph.

Uses the tensorrt Python bindings (TRT 11.2, sm_120 builder resources included in
tensorrt_libs). Produces a .plan engine + build metadata JSON. INT8 is a later,
accuracy-gated step (see bench_trt.py).

Usage (WSL, egohand env):
  LD_LIBRARY_PATH=<tensorrt_libs>:<nvidia libs> \
    python optimize/build_trt_engine.py --onnx results/onnx/wilor-fast.onnx \
      --out results/trt/wilor-fast-fp16.plan
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(ROOT / "results" / "onnx" / "wilor-fast.onnx"))
    ap.add_argument("--out", default=str(ROOT / "results" / "trt" / "wilor-fast-fp16.plan"))
    ap.add_argument("--workspace-gb", type=float, default=3.0)
    args = ap.parse_args()

    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    onnx_path = Path(args.onnx)
    # External-data weights live next to the .onnx (Constant_*_attr__value files).
    ok = parser.parse_from_file(str(onnx_path))
    if not ok:
        for i in range(parser.num_errors):
            print("parse error:", parser.get_error(i))
        sys.exit("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb * 1e9))
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 flag: enabled")
    else:
        print("FP16 not supported by platform — building FP32 (unexpected for sm_120)")

    profile = builder.create_optimization_profile()
    # Static shape graph: input declared at build time as-is.
    profile.set_shape("image", (1, 3, 256, 256), (1, 3, 256, 256), (1, 3, 256, 256))
    config.add_optimization_profile(profile)

    print("building engine (this can take several minutes)...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        sys.exit("engine build failed")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = bytes(engine_bytes) if not isinstance(engine_bytes, (bytes, bytearray)) else engine_bytes
    out.write_bytes(data)
    meta = {
        "onnx": str(onnx_path),
        "precision": "fp16",
        "workspace_gb": args.workspace_gb,
        "engine_bytes": len(data),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "tensorrt": trt.__version__,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"engine written: {out} ({len(data) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
