"""Export the fixed-shape WiLoR reconstruction graph.

Dynamic YOLO detection and ViTDetDataset crop generation remain outside ONNX. The graph
receives one normalized 256x256 crop plus a right-hand scalar, matching the official
reconstruction input contract. This is more deployable and exportable than attempting to
put detector/NMS/cropping and Python dicts into one graph.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "results" / "onnx" / "winner.onnx"))
    p.add_argument("--variant", choices=["default", "fast"], default="fast")
    args = p.parse_args()

    # Import wrapper only inside this env; WILOR_ROOT is required.
    from models.wilor_wrapper import WiLoRHandModel
    model = WiLoRHandModel(variant=args.variant)
    model.load("cuda")
    inner = model.model
    # torch.compile (fast variant) is incompatible with torch.jit.trace — unwrap
    # to the eager module for export; FP16 weights are preserved.
    eager_backbone = getattr(inner.backbone, "_orig_mod", None)
    if eager_backbone is not None:
        inner.backbone = eager_backbone
    if args.variant == "fast":
        # ONNX export needs a normal eager module; torch.compile is not serializable here.
        # Retain FP16 weights/inputs, but bypass the official fast skip_blocks toggle for
        # export correctness unless the exporter accepts it.
        inner.backbone.skip_blocks = False

    # Trace a plain function, NOT the LightningModule: torch.jit.trace inspects
    # submodules and Lightning's `trainer` property raises on hasattr probing.
    dtype = next(inner.parameters()).dtype
    device = next(inner.parameters()).device
    dummy = torch.zeros(1, 3, 256, 256, device=device, dtype=dtype)
    right = torch.ones(1, device=device, dtype=dtype)

    def graph_forward(image: torch.Tensor, right_: torch.Tensor) -> tuple:
        b = image.shape[0]
        out = inner.forward_step(
            {"img": image, "right": right_.reshape(b)}, train=False
        )
        return out["pred_keypoints_3d"], out["pred_vertices"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Freeze everything: under no_grad, a requires_grad tensor captured as a
    # trace constant triggers "Cannot insert a Tensor that requires grad".
    inner.requires_grad_(False)

    # Trace the plain function, then export the traced graph.
    with torch.no_grad():
        traced = torch.jit.trace(graph_forward, (dummy, right), strict=False)
    torch.onnx.export(
        traced, (dummy, right), str(out_path),
        export_params=True, opset_version=17,
        input_names=["image", "right"],
        output_names=["joints_3d", "mesh_verts"],
        dynamic_axes=None,
        dynamo=False,
    )
    meta = {"variant": args.variant, "input_shape": [1, 3, 256, 256],
            "outputs": {"joints_3d": [1, 21, 3], "mesh_verts": [1, 778, 3]},
            "detector_and_crop_outside_graph": True}
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"exported {out_path}")


if __name__ == "__main__":
    main()