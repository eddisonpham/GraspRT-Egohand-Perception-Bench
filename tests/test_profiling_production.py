"""Unit tests for common/profiling.py production monitoring helpers.

Covers nvidia_smi_snapshot, cpu_load_sample, and ResourceMonitor with mocked
subprocess calls (no real GPU needed on CI host).

Run: python -m pytest tests/test_profiling_production.py -q
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.profiling import (  # noqa: E402
    ResourceMonitor,
    cpu_load_sample,
    nvidia_smi_snapshot,
)


# ── nvidia_smi_snapshot ───────────────────────────────────────────────

def test_snapshot_returns_dict_with_expected_keys():
    """When nvidia-smi is available, snapshot must return all four keys."""
    fake_output = "78, 4235, 65.42, 52\n"
    with patch("subprocess.run", return_value=subprocess.CompletedProcess(
        args=[], returncode=0, stdout=fake_output, stderr="")):
        snap = nvidia_smi_snapshot()
    assert set(snap) == {"gpu_util_pct", "mem_used_mb", "power_watts", "temp_c"}
    assert snap["gpu_util_pct"] == 78.0
    assert snap["mem_used_mb"] == 4235.0
    assert snap["power_watts"] == 65.42
    assert snap["temp_c"] == 52.0


def test_snapshot_returns_empty_on_nvidia_smi_failure():
    """When nvidia-smi is missing or errors, snapshot must not raise."""
    with patch("subprocess.run", side_effect=FileNotFoundError("nvidia-smi")):
        assert nvidia_smi_snapshot() == {}


def test_snapshot_returns_empty_on_malformed_output():
    with patch("subprocess.run", return_value=subprocess.CompletedProcess(
        args=[], returncode=0, stdout="", stderr="")):
        assert nvidia_smi_snapshot() == {}


def test_snapshot_handles_na_values():
    """Driver may report N/A when GPU is idle."""
    fake_output = "[N/A], N/A, 0.00, 0\n"
    with patch("subprocess.run", return_value=subprocess.CompletedProcess(
        args=[], returncode=0, stdout=fake_output, stderr="")):
        snap = nvidia_smi_snapshot()
    assert snap["gpu_util_pct"] == 0.0
    assert snap["mem_used_mb"] == 0.0


def test_snapshot_handles_negative_power():
    """Some drivers report -1 power on desktop GPUs without sensors."""
    fake_output = "45, 1024, -1, 40\n"
    with patch("subprocess.run", return_value=subprocess.CompletedProcess(
        args=[], returncode=0, stdout=fake_output, stderr="")):
        snap = nvidia_smi_snapshot()
    assert snap["power_watts"] == -1.0  # no clamping; caller decides


# ── cpu_load_sample ───────────────────────────────────────────────────

def test_cpu_load_has_load_1m():
    """On Linux/WSL, os.getloadavg is available; load_1m must be numeric."""
    snap = cpu_load_sample()
    if "load_1m" in snap:
        assert isinstance(snap["load_1m"], float)
        assert snap["load_1m"] >= 0.0


def test_cpu_load_has_rss():
    """On Linux, /proc/self/status provides VmRSS."""
    snap = cpu_load_sample()
    if "process_rss_mb" in snap:
        assert snap["process_rss_mb"] > 0.0


def test_cpu_load_never_raises():
    """cpu_load_sample must never throw regardless of environment."""
    import os
    has_loadavg = hasattr(os, "getloadavg")
    # Force os.getloadavg to be unavailable to test fallback path.
    if has_loadavg:
        orig = os.getloadavg
        del os.getloadavg
    try:
        snap = cpu_load_sample()
        assert isinstance(snap, dict)
    finally:
        if has_loadavg:
            os.getloadavg = orig


# ── ResourceMonitor ───────────────────────────────────────────────────

def test_resource_monitor_stops_cleanly():
    """Monitor starts, runs a few intervals, and stops without hanging."""
    mon = ResourceMonitor(interval_s=0.02)
    mon.start()
    time.sleep(0.12)
    mon.stop()
    assert not mon.is_alive()


def test_resource_monitor_collects_samples():
    """With a mocked snapshot, monitor should collect multiple samples."""
    fake_snap = {"gpu_util_pct": 50.0, "mem_used_mb": 2000.0,
                 "power_watts": 45.0, "temp_c": 42.0, "load_1m": 1.5,
                 "process_rss_mb": 800.0}
    with patch("common.profiling.nvidia_smi_snapshot", return_value=fake_snap), \
         patch("common.profiling.cpu_load_sample", return_value={"load_1m": 1.5}):
        mon = ResourceMonitor(interval_s=0.02)
        mon.start()
        time.sleep(0.1)
        mon.stop()
    assert len(mon.samples) >= 3
    assert mon.peak["gpu_util_pct"] == 50.0
    assert mon.peak["load_1m"] == 1.5


def test_resource_monitor_summary_schema():
    """summary() must return n_samples + per-metric _peak and _mean."""
    fake_snap = {"gpu_util_pct": 60.0, "power_watts": 70.0}
    with patch("common.profiling.nvidia_smi_snapshot", return_value=fake_snap), \
         patch("common.profiling.cpu_load_sample", return_value={}):
        mon = ResourceMonitor(interval_s=0.02)
        mon.start()
        time.sleep(0.1)
        mon.stop()
    s = mon.summary()
    assert s["n_samples"] >= 3
    assert "gpu_util_pct_peak" in s
    assert "gpu_util_pct_mean" in s
    assert "power_watts_peak" in s
    assert s["gpu_util_pct_peak"] == 60.0


def test_resource_monitor_empty_summary():
    """summary() on a never-started monitor returns n_samples=0."""
    mon = ResourceMonitor()
    s = mon.summary()
    assert s == {"n_samples": 0}
