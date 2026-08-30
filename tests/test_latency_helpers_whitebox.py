"""White-box tests for the latency summarize/pct helpers in optimize scripts.

Targets the percentile helper directly: boundary indices, small-n rounding,
and the summarize() contract (mean/p50/p95/p99/fps/n). Run:
    python -m pytest tests/test_latency_helpers_whitebox.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optimize.bench_onnx_latency import pct, summarize  # noqa: E402


def test_pct_extremes():
    v = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert pct(v, 0) == 10.0
    assert pct(v, 100) == 50.0
    assert pct(v, 50) == 30.0


def test_pct_sorted_independently():
    unsorted = [50.0, 10.0, 40.0, 20.0, 30.0]
    assert pct(unsorted, 95) == pct(sorted(unsorted), 95)


def test_pct_single_element():
    assert pct([7.0], 0) == 7.0
    assert pct([7.0], 99) == 7.0


def test_pct_clamps_out_of_range():
    # p beyond [0,100] should clamp, not crash or extrapolate.
    v = [1.0, 2.0, 3.0]
    assert pct(v, -5) == 1.0
    assert pct(v, 250) == 3.0


def test_summarize_contract():
    s = summarize([10.0, 20.0, 30.0, 40.0])
    assert s["n"] == 4
    assert s["mean_ms"] == 25.0
    assert s["fps"] == round(1000 / 25.0, 2)  # 40.0
    # all expected keys present
    assert {"mean_ms", "p50_ms", "p95_ms", "p99_ms", "fps", "n"} == set(s)


def test_summarize_p50_equals_median():
    s = summarize([5.0, 1.0, 3.0, 2.0, 4.0])
    assert s["p50_ms"] == 3.0  # median of sorted [1,2,3,4,5]


def test_summarize_p99_le_p95_for_small_n_is_valid():
    # with small n, p99 and p95 may coincide; just ensure they're consistent.
    s = summarize([1.0, 2.0, 3.0])
    assert s["p95_ms"] <= s["p99_ms"] + 1e-9 or s["p95_ms"] == s["p99_ms"]


def test_pct_p95_typical_latency_run():
    # 100 samples 1..100; p95 should be ~95-96 (nearest-rank style).
    v = [float(i) for i in range(1, 101)]
    p95 = pct(v, 95)
    assert 94 <= p95 <= 96, f"p95={p95}"
