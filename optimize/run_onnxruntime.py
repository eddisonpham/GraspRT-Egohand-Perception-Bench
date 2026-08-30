"""ONNX Runtime benchmark entrypoint.

This intentionally refuses to benchmark a missing/unverified graph. It can be used once the
MANO-boundary graph split described in optimization_notes.md is implemented.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import onnxruntime as ort


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", default="results/onnx/winner.onnx")
    args = p.parse_args()
    path = Path(args.onnx)
    if not path.exists():
        raise FileNotFoundError(f"No verified ONNX graph at {path}; export must pass first")
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    print("onnx checker: PASS")
    print("available providers:", ort.get_available_providers())
    for providers in [["CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"], ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]]:
        try:
            sess = ort.InferenceSession(str(path), providers=providers)
            print("provider request:", providers, "active:", sess.get_providers())
        except Exception as exc:
            print("provider request FAILED:", providers, repr(exc))


if __name__ == "__main__":
    main()