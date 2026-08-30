"""Shared latency and resource profiling.

One timing protocol across every backend (PyTorch / ONNX Runtime / TensorRT)
so benchmark rows are comparable. GPU latency uses CUDA events; CPU models use
perf_counter. VRAM is tracked via torch allocator + nvidia-smi.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from typing import Optional

import numpy as np


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def time_infer(model, batch, n_warmup: int = 20, n_iters: int = 200,
               use_cuda_events: Optional[bool] = None) -> dict:
    """Time model.infer(batch); returns mean/median/p95/std ms (warmup excluded)."""
    if use_cuda_events is None:
        use_cuda_events = (getattr(model, "device", "cpu") != "cpu") and _cuda_available()

    for _ in range(max(1, n_warmup)):
        model.infer(batch)

    if use_cuda_events:
        import torch
        torch.cuda.synchronize()
        lat = []
        for _ in range(n_iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
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
            lat.append((time.perf_counter() - t0) * 1e3)

    arr = np.asarray(lat, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "std_ms": float(arr.std()),
    }


def torch_peak_mb() -> float:
    """Peak VRAM reported by PyTorch's allocator."""
    import torch
    return torch.cuda.max_memory_allocated() / 1e6


def _smi_query(query: str) -> str:
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    )
    return out.stdout.strip()


def nvidia_smi_used_mb() -> float:
    return float(_smi_query("memory.used").splitlines()[0])


def nvidia_smi_snapshot() -> dict:
    """Sample GPU util/VRAM/power/temp. Returns {} if nvidia-smi is unavailable."""
    try:
        raw = _smi_query("utilization.gpu,memory.used,power.draw,temperature.gpu")
        parts = [p.strip() for p in raw.splitlines()[0].split(",")]
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
    """Snapshot host load average + process RSS (best-effort, never raises)."""
    snap: dict = {}
    try:
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
    """Background thread sampling GPU util/power/temp + CPU load every interval_s.

    mon.summary() returns per-metric peak and mean over the sampled window.
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
        """Per-metric peak and mean over the sampled window."""
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
    """Background thread polling nvidia-smi memory.used every interval_s."""

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
    """Measure torch-allocator AND nvidia-smi peak across a timed run."""
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
    """Write results/raw/<model_name>.json with the standard schema metadata."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{model_name}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path