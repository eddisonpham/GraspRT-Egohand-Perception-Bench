"""TRT YOLO backbone+neck (layers 0..21) + native Pose head.

Replicates ultralytics _predict_once faithfully (honoring m.f / y[routes])
to produce the 3 feature maps layers 15/18/21 that the Pose head consumes.
Exports these to TRT FP16, then runs native Pose head + NMS on top.

If boxes match torch to <0.1px, this preserves accuracy while accelerating
the dominant 13.8ms conv forward.
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
    import torch

    from data.freihand.loader import FreiHandLoader
    from models.wilor_wrapper import WiLoRHandModel

    model = WiLoRHandModel(variant="fast")
    model.load("cuda")
    det = model.detector
    full_model = det.model
    seq = full_model.model  # 23 layers, last is Pose

    loader = FreiHandLoader(subset="data/freihand/subsets/dev.json")
    images = [loader[i][0] for i in range(30)]
    pred = det.predictor

    # --- Baseline torch boxes ---
    for img in images[:5]:
        det(img, verbose=False)
    torch_boxes = []
    for img in images:
        r = det(img, verbose=False)[0]
        torch_boxes.append(r.boxes.xyxy.detach().cpu().numpy() if r.boxes is not None else None)

    # --- Faithful routing wrapper: run layers 0..21, capture y[15], y[18], y[21] ---
    save = set(getattr(full_model, "save", set())) | {15, 18, 21}

    class BackboneRoutes(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.ModuleList(seq[:22])

        def forward(self, x):
            y = [None] * 23
            for i, mod in enumerate(self.layer):
                if mod.f != -1:
                    src = mod.f
                    x = y[src] if isinstance(src, int) else \
                        [x if j == -1 else y[j] for j in src]
                x = mod(x)
                y[i] = x if i in save else None
            return tuple(y[j] for j in (15, 18, 21))

    engine_path = str(ROOT / "results" / "trt" / "yolo-backbone-fp16.engine")
    if not os.path.exists(engine_path):
        routes = BackboneRoutes().float().cuda().eval()
        dummy = torch.randn(1, 3, 224, 224, device="cuda", dtype=torch.float32)
        onnx_path = str(ROOT / "results" / "onnx" / "yolo-backbone.onnx")
        print("Exporting backbone routes to ONNX ...")
        torch.onnx.export(routes, dummy, onnx_path, dynamo=False,
                          input_names=["image"], output_names=["p3", "p4", "p5"],
                          opset_version=17, do_constant_folding=True)
        print("ONNX done")
        del routes, dummy
        torch.cuda.empty_cache()

        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(onnx_path):
            for i in range(parser.num_errors):
                print("  parse err:", parser.get_error(i))
            sys.exit("backbone parse failed")
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(3e9))
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        prof = builder.create_optimization_profile()
        shape = (1, 3, 224, 224)
        prof.set_shape("image", shape, shape, shape)
        config.add_optimization_profile(prof)
        print("Building backbone TRT engine ...")
        t0 = time.perf_counter()
        plan = builder.build_serialized_network(network, config)
        print(f"built in {time.perf_counter()-t0:.1f}s")
        Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
        Path(engine_path).write_bytes(bytes(plan))
        print("saved", engine_path)

    # --- Load engine + run native Pose head = full detector ---
    import tensorrt as trt
    from ultralytics.utils.nms import non_max_suppression

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        eng = runtime.deserialize_cuda_engine(f.read())
    ctx = eng.create_execution_context()
    pose_head = seq[22]

    # ensure predictor is initialized and Pose head anchors/strides are primed
    for img in images[:3]:
        det(img, verbose=False)
    pred = det.predictor
    assert pred is not None
    _ = det.model(torch.randn(1, 3, 224, 224, device="cuda"))  # prime head anchors

    # probe feature shapes via torch routing
    routes_t = BackboneRoutes().float().cuda().eval()
    with torch.no_grad():
        feats_probe = routes_t(torch.randn(1, 3, 224, 224, device="cuda"))
    d_out = []
    for f in feats_probe:
        d_out.append(torch.empty_like(f).half())
    print("feature shapes:", [tuple(f.shape) for f in feats_probe])

    trt_boxes = []
    times = []
    for img in images:
        im = pred.preprocess([img])
        t0 = time.perf_counter()
        hi = im.half()
        ctx.execute_v2(bindings=[hi.data_ptr()] + [o.data_ptr() for o in d_out])
        torch.cuda.synchronize()
        out_feats = tuple(f.float() for f in d_out)
        dec = pose_head.forward(out_feats)
        x = non_max_suppression(dec, conf_thres=0.3, iou_thres=0.45)[0]
        times.append((time.perf_counter() - t0) * 1000)
        trt_boxes.append(x[:, :4].detach().cpu().numpy() if x is not None and len(x) else None)

    diffs = []
    for tb, db in zip(torch_boxes, trt_boxes):
        if tb is None or db is None:
            continue
        if len(tb) == 0 or len(db) == 0:
            continue
        diffs.append(float(np.abs(tb[:1] - db[:1]).max()))
    mean_diff = float(np.mean(diffs)) if diffs else 999
    arr = np.asarray(times)
    print("\n=== TRT backbone+neck + NATIVE Pose head ===")
    print(f"  latency mean {arr.mean():.3f} ms  p95 {np.percentile(arr,95):.3f}")
    print(f"  mean max-abs box diff vs torch: {mean_diff:.3f} px (n={len(diffs)})")

    out = ROOT / "results" / "raw" / "yolo-backbone-trt.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "engine": engine_path,
        "latency_mean_ms": round(float(arr.mean()), 3),
        "latency_p95_ms": round(float(np.percentile(arr, 95)), 3),
        "mean_max_box_diff_px": round(mean_diff, 3),
        "n_pairs": len(diffs),
    }, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()