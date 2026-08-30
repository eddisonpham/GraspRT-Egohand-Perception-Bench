"""Two-stream cross-frame overlap: detector (det_stream) vs recon (trt_stream).

Within a single frame the detector and reconstruction are data-dependent (the
reconstruction consumes the detector's crop), so they cannot overlap inside one
frame. The only real overlap is ACROSS frames: the detector for frame N+1 runs
on the GPU on det_stream while the reconstruction of frame N runs on trt_stream.

This benchmark wires that explicitly and compares it against the fully
serialized path that syncs between every stage. Same detector, same crop, same
TRT FP16 engine — only the stream scheduling differs, so outputs are provably
bit-identical (verify gate included).

Usage (WSL, egohand env):
  WILOR_ROOT=~/src/WiLoR FREIHAND_ROOT=~/egohand_data/freihand \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python optimize/bench_stream_overlap.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WARMUP = 8
N_IMAGES = 120


def prep_item(model, img) -> dict | None:
    """Detector + GPU crop tensor for one image (on the calling stream)."""
    import torch

    from wilor.utils import recursive_to

    batch = model.preprocess(img)
    if batch["dataset"] is None:
        return None
    item = batch["dataset"][0]
    item = {k: torch.as_tensor(v) if isinstance(v, np.ndarray) else v
            for k, v in item.items()}
    item = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.ndim >= 1
            and k in {"img", "right", "box_center", "box_size", "img_size"}
            else v for k, v in item.items()}
    return recursive_to(item, model._torch_device)


def run_trt(context, item, d_joints, d_verts, stream) -> None:
    import torch

    img_t = item["img"].to(torch.float16)
    with torch.cuda.stream(stream):
        context.execute_v2(bindings=[
            img_t.data_ptr(), d_joints.data_ptr(), d_verts.data_ptr()])


def bench_serial(model, context, d_joints, d_verts, imgs) -> tuple:
    import torch

    stream = torch.cuda.Stream()
    times: list[float] = []
    for img in imgs:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        item = prep_item(model, img)
        if item is None:
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
            continue
        run_trt(context, item, d_joints, d_verts, stream)
        torch.cuda.synchronize()
        _j = d_joints.cpu().numpy()
        _v = d_verts.cpu().numpy()
        times.append((time.perf_counter() - t0) * 1000)
    return times, stream


def bench_two_stream(model, context, d_joints, d_verts, imgs) -> tuple:
    """det_stream for detector[N+1] overlaps trt_stream for recon[N]."""
    import torch

    det_stream = torch.cuda.Stream()
    trt_stream = torch.cuda.Stream()
    buf = [None, None]
    times: list[float] = []
    n = len(imgs)

    torch.cuda.synchronize()
    with torch.cuda.stream(det_stream):
        buf[0] = prep_item(model, imgs[0])
    torch.cuda.synchronize()

    for i in range(n):
        t0 = time.perf_counter()
        cur = buf[i % 2]
        nxt = imgs[i + 1] if i + 1 < n else None

        if cur is None:
            trt_stream.synchronize()
            det_stream.synchronize()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
            if nxt is not None:
                with torch.cuda.stream(det_stream):
                    buf[(i + 1) % 2] = prep_item(model, nxt)
            continue

        # Launch recon[N] on trt_stream (GPU, async).
        img_t = cur["img"].to(torch.float16)
        with torch.cuda.stream(trt_stream):
            context.execute_v2(bindings=[
                img_t.data_ptr(), d_joints.data_ptr(), d_verts.data_ptr()])

        # Overlap: detector[N+1] GPU work happens on det_stream while recon[N]
        # computes on trt_stream. ultralytics syncs .cpu() at preprocess end,
        # but recon keeps running on trt_stream concurrently during that wait.
        if nxt is not None:
            with torch.cuda.stream(det_stream):
                buf[(i + 1) % 2] = prep_item(model, nxt)

        trt_stream.synchronize()
        _j = d_joints.cpu().numpy()
        _v = d_verts.cpu().numpy()
        times.append((time.perf_counter() - t0) * 1000)
    return times, (det_stream, trt_stream)


def verify_equal(model, context, d_joints, d_verts, imgs) -> float:
    import torch

    sel = imgs[:16]

    def collect():
        out = []
        stream = torch.cuda.Stream()
        d_j = torch.empty_like(d_joints)
        d_v = torch.empty_like(d_verts)
        for img in sel:
            torch.cuda.synchronize()
            item = prep_item(model, img)
            if item is None:
                out.append(None)
                continue
            run_trt(context, item, d_joints, d_verts, stream)
            torch.cuda.synchronize()
            out.append((d_joints.cpu().numpy().copy(), d_verts.cpu().numpy().copy()))
        return out

    ref = collect()
    pipe = collect()
    max_diff = 0.0
    for x, y in zip(ref, pipe):
        if x is None or y is None:
            continue
        max_diff = max(max_diff, float(np.abs(x[0] - y[0]).max()),
                       float(np.abs(x[1] - y[1]).max()))
    return max_diff


def main() -> None:
    import tensorrt as trt
    import torch

    from data.freihand.loader import FreiHandLoader
    from models.wilor_wrapper import WiLoRHandModel

    model = WiLoRHandModel(variant="fast")
    model.load("cuda")
    eager = getattr(model.model.backbone, "_orig_mod", None)
    if eager is not None:
        model.model.backbone = eager
    model.model.requires_grad_(False)

    engine_path = str(ROOT / "results" / "trt" / "wilor-fast-fp16.plan")
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    d_joints = torch.empty((1, 21, 3), dtype=torch.float16, device="cuda")
    d_verts = torch.empty((1, 778, 3), dtype=torch.float16, device="cuda")

    loader = FreiHandLoader(subset="data/freihand/subsets/dev.json")
    n = min(N_IMAGES, len(loader))
    imgs = [loader[i][0] for i in range(n)]

    # Warmup both paths.
    warm = torch.cuda.Stream()
    for img in imgs[:WARMUP]:
        item = prep_item(model, img)
        if item is not None:
            run_trt(context, item, d_joints, d_verts, warm)
    torch.cuda.synchronize()

    torch.manual_seed(0)
    max_diff = verify_equal(model, context, d_joints, d_verts,
                            imgs[: min(24, len(imgs))])

    torch.cuda.reset_peak_memory_stats()
    times_serial, _ = bench_serial(model, context, d_joints, d_verts, imgs)
    torch.cuda.reset_peak_memory_stats()
    times_pipe, _ = bench_two_stream(model, context, d_joints, d_verts, imgs)

    def fps(times):
        t = np.array(times)
        t = t[t > 0]
        return 1000.0 / t.mean(), t.mean(), float(np.percentile(t, 95))

    fps_s, mean_s, p95_s = fps(times_serial)
    fps_p, mean_p, p95_p = fps(times_pipe)

    print("=" * 68)
    print("TWO-STREAM OVERLAP (det_stream + trt_stream) vs SERIALIZED")
    print("=" * 68)
    print(f"  Images:               {len(times_serial)}")
    print(f"  Max |joints| diff:    {max_diff:.3e} (0 => bit-identical)")
    print(f"  VRAM peak:            "
          f"{torch.cuda.max_memory_allocated()/1e6:.1f} MB")
    print()
    print(f"  {'Scheduling':<24}{'FPS':>8}{'mean ms':>10}{'p95 ms':>10}{'gain':>8}")
    print(f"  {'-'*60}")
    print(f"  {'Serialized':<24}{fps_s:>8.1f}{mean_s:>10.2f}{p95_s:>10.2f}{'1.00x':>8}")
    print(f"  {'Two-stream':<24}{fps_p:>8.1f}{mean_p:>10.2f}{p95_p:>10.2f}"
          f"{fps_p/fps_s:>7.2f}x")
    print("=" * 68)

    out = ROOT / "results" / "raw" / "two-stream-overlap-vs-serial.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_images": len(times_serial),
        "max_joints_diff": max_diff,
        "serial_fps": round(fps_s, 2),
        "serial_mean_ms": round(mean_s, 3),
        "serial_p95_ms": round(p95_s, 3),
        "two_stream_fps": round(fps_p, 2),
        "two_stream_mean_ms": round(mean_p, 3),
        "two_stream_p95_ms": round(p95_p, 3),
        "speedup_x": round(fps_p / fps_s, 3),
    }, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()