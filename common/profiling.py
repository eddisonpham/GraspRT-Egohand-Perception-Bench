"""Latency + VRAM profiling utilities.

Design intent (stage 06): every benchmark run, across every model and every backend
(PyTorch, ONNX Runtime, TensorRT), must use the SAME warmup/timing protocol so rows in
the final tables are comparable. This module is the single implementation of that protocol.

- GPU-side latency: CUDA events (host dispatch overhead excluded); CPU-only models
  (MediaPipe) use time.perf_counter().
- VRAM: torch allocator peak + nvidia-smi poll (driver-level truth, includes context
  overhead and anything outside PyTorch's allocator).
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
from typing import Optional

import numpy as np


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def time_infer(model, batch, n_warmup: int = 20, n_iters: int = 200,
               use_cuda_events: Optional[bool] = None) -> dict:
    """Time model.infer(batch) using the project's standard protocol.

    Returns {"mean_ms","median_ms","p95_ms","std_ms"}. Warmup happens first and is
    excluded from all numbers.
    """
    if use_cuda_events is None:
        use_cuda_events = (getattr(model, "device", "cpu") != "cpu") and _cuda_available()

    # warmup — excluded from timing
    for _ in range(max(1, n_warmup)):
        model.infer(batch)

    if use_cuda_events:
        import torch
        torch.cuda.synchronize()
        lat = []
        for _ in range(n_iters):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            model.infer(batch)
            end.record()
            torch.cuda.synchronize()
            lat.append(start.elapsed_time(end))
    else:
        lat = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            model.infer(batch)
            t1 = time.perf_counter()
            lat.append((t1 - t0) * 1e3)

    arr = np.asarray(lat, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "std_ms": float(arr.std()),
    }


def torch_peak_mb() -> float:
    """Peak VRAM as reported by PyTorch's allocator (requires torch+cuda in env)."""
    import torch
    return torch.cuda.max_memory_allocated() / 1e6


def nvidia_smi_used_mb() -> float:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    )
    return float(out.stdout.strip().splitlines()[0])


def nvidia_smi_snapshot() -> dict:
    """One driver-level sample of GPU utilization, memory, power, and temperature.

    Returns {} when nvidia-smi is unavailable (non-NVIDIA host) so callers can
    treat production metrics as optional rather than hard-failing.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
        if len(parts) < 4:
            return {}
        def _num(s: str) -> float:
            try:
                return float(s.replace("[N/A]", "0").replace("N/A", "0"))
            except ValueError:
                return 0.0
        return {
            "gpu_util_pct": _num(parts[0]),
            "mem_used_mb": _num(parts[1]),
            "power_watts": _num(parts[2]),
            "temp_c": _num(parts[3]),
        }
    except Exception:
        return {}


def cpu_load_sample() -> dict:
    """Host CPU load + process RSS snapshot (cross-platform best-effort).

    Uses os.getloadavg (POSIX) and /proc/self/status VmRSS (Linux). On hosts
    without either, returns whatever is available — never raises.
    """
    snap: dict = {}
    try:
        import os
        if hasattr(os, "getloadavg"):
            la = os.getloadavg()
            snap["load_1m"] = round(float(la[0]), 3)
            snap["load_5m"] = round(float(la[1]), 3)
    except Exception:
        pass
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    snap["process_rss_mb"] = round(float(line.split()[1]) / 1024.0, 1)
                    break
    except Exception:
        pass
    return snap


class ResourceMonitor(threading.Thread):
    """Background production-profiling thread: GPU + CPU + power samples.

    Extends NvidiaSmiMonitor with utilization/power/temperature and host CPU
    load, all sampled every `interval_s`. `peak` keeps the max per metric and
    `samples` keeps the full time series for after-run analysis.

    Usage:
        mon = ResourceMonitor()
        mon.start()
        ... workload ...
        mon.stop()
        mon.peak  # {"gpu_util_pct": ..., "power_watts": ..., ...}
    """

    def __init__(self, interval_s: float = 0.05):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self.peak: dict[str, float] = {}
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            snap = {**nvidia_smi_snapshot(), **cpu_load_sample()}
            if snap:
                self.samples.append(snap)
                for k, v in snap.items():
                    self.peak[k] = max(self.peak.get(k, 0.0), v)
            self._stop_event.wait(self.interval_s)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=2.0)

    def summary(self) -> dict:
        """Peak + mean per metric over the sampled window."""
        if not self.samples:
            return {"n_samples": 0}
        keys = set().union(*(s.keys() for s in self.samples))
        out: dict = {"n_samples": len(self.samples)}
        for k in sorted(keys):
            vals = [s[k] for s in self.samples if k in s]
            if vals:
                out[f"{k}_peak"] = round(max(vals), 3)
                out[f"{k}_mean"] = round(sum(vals) / len(vals), 3)
        return out


class NvidiaSmiMonitor(threading.Thread):
    """Background thread polling nvidia-smi memory.used every `interval_s` seconds.

    Usage:
        mon = NvidiaSmiMonitor()
        mon.start()
        ... run workload ...
        mon.stop()
        peak = mon.peak_mb
    """

    def __init__(self, interval_s: float = 0.05):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.peak_mb = 0.0
        self.samples_mb: list[float] = []
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            try:
                mb = nvidia_smi_used_mb()
                self.samples_mb.append(mb)
                self.peak_mb = max(self.peak_mb, mb)
            except Exception:
                pass
            self._stop_event.wait(self.interval_s)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=2.0)


def measure_vram(model, batch, n_warmup: int = 20, n_iters: int = 200) -> dict:
    """Measure torch-allocator peak AND nvidia-smi peak across a timed run."""
    import torch
    torch.cuda.reset_peak_memory_stats()
    mon = NvidiaSmiMonitor()
    mon.start()
    try:
        time_infer(model, batch, n_warmup=n_warmup, n_iters=n_iters)
    finally:
        mon.stop()
    return {
        "torch_peak": torch_peak_mb(),
        "nvidia_smi_peak": mon.peak_mb,
        "nvidia_smi_samples": mon.samples_mb,
    }


def write_raw(model_name: str, variant: str, payload: dict, out_dir="results/raw") -> str:
    """Write one results/raw/<model>.json in the stage-06 schema.

    Accepts a payload dict; merges required key metadata. Keeps the schema in one place.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{model_name}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path