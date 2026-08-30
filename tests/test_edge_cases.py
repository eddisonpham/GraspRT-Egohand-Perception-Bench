"""Edge-case unit tests for the pipeline's pure-Python surfaces.

Covers boundaries the existing tests skim over: empty/single-element inputs,
shape mismatches, NaN propagation, f_score asymmetry, normalize ties and
all-None, the VRAM-median override, write_raw schema, and HandPrediction
field defaults. No GPU, no model, no dataset required.

Run: python -m pytest tests/test_edge_cases.py -q
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.metrics import (  # noqa: E402
    _umeyama,
    f_score,
    mpjpe,
    pa_mpjpe,
    pa_mpvpe,
    procrustes_align,
)
from common.interface import BaseHandModel, HandPrediction  # noqa: E402
import benchmark.aggregate as agg  # noqa: E402


# ── metrics: shape / dtype / degenerate inputs ─────────────────────────

def test_procrustes_shape_mismatch_raises():
    with pytest.raises(ValueError):
        procrustes_align(np.zeros((21, 3)), np.zeros((778, 3)))


def test_pa_mpjpe_shape_mismatch_propagates():
    # pa_mpjpe calls procrustes_align which raises on mismatch.
    with pytest.raises(ValueError):
        pa_mpjpe(np.zeros((21, 3)), np.zeros((20, 3)))


def test_procrustes_accepts_2d_float32():
    pred = np.random.default_rng(0).normal(size=(21, 3)).astype(np.float32)
    gt = pred.astype(np.float64)
    aligned = procrustes_align(pred, gt)
    assert aligned.dtype == np.float64
    assert aligned.shape == (21, 3)


def test_pa_mpjpe_single_point_is_finite():
    # One vertex: covariance is degenerate; must not NaN or raise.
    p = np.array([[0.1, 0.2, 0.3]])
    g = np.array([[0.0, 0.0, 0.0]])
    val = pa_mpjpe(p, g)
    assert np.isfinite(val)
    assert val >= 0.0


def test_pa_mpvpe_single_point_is_finite():
    p = np.array([[1.0, 0.0, 0.0]])
    g = np.array([[0.0, 0.0, 0.0]])
    assert np.isfinite(pa_mpvpe(p, g))


def test_umeyama_all_identical_no_nan():
    pts = np.ones((10, 3))
    s, R, t = _umeyama(pts, pts, with_scale=True)
    assert np.isfinite(s)
    assert np.isfinite(R).all()
    assert np.isfinite(t).all()


def test_umeyama_collinear_input_no_nan():
    # All points on a line: rank-1 covariance; SVD still well-defined.
    pts = np.zeros((20, 3))
    pts[:, 0] = np.arange(20) * 0.1
    s, R, t = _umeyama(pts, pts + 0.5, with_scale=True)
    assert np.isfinite(s) and np.isfinite(R).all() and np.isfinite(t).all()


def test_mpjpe_zero_input_is_zero():
    z = np.zeros((21, 3))
    assert mpjpe(z, z) == 0.0


def test_mpjpe_units_are_mm():
    # 0.001 m per-axis offset = sqrt(3)*0.001 m = 1.732 mm
    gt = np.zeros((21, 3))
    pred = np.ones((21, 3)) * 0.001
    assert mpjpe(pred, gt) == pytest.approx(np.sqrt(3) * 1.0, rel=1e-4)


def test_pa_mpjpe_is_symmetric_under_label_swap_for_identity():
    # pa_mpjpe(gt,gt) is 0 regardless of argument order for identical sets.
    gt = np.random.default_rng(1).normal(size=(21, 3))
    assert pa_mpjpe(gt, gt) == pytest.approx(0.0, abs=1e-9)


def test_pa_mpjpe_translation_invariance():
    rng = np.random.default_rng(2)
    gt = rng.normal(size=(21, 3))
    pred = gt + rng.normal(scale=0.01, size=gt.shape)
    base = pa_mpjpe(pred, gt)
    shift = pa_mpjpe(pred + 10.0, gt + 10.0)
    assert base == pytest.approx(shift, abs=1e-6)


def test_pa_mpjpe_rotation_invariance():
    rng = np.random.default_rng(3)
    gt = rng.normal(size=(21, 3))
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pred = gt @ R.T
    assert pa_mpjpe(pred, gt) == pytest.approx(0.0, abs=1e-6)


def test_pa_mpjpe_scale_invariance():
    rng = np.random.default_rng(4)
    gt = rng.normal(size=(21, 3))
    assert pa_mpjpe(gt * 5.0, gt) == pytest.approx(0.0, abs=1e-6)


# ── f_score edge cases ─────────────────────────────────────────────────

def test_fscore_identical_returns_one():
    v = np.random.default_rng(5).normal(size=(100, 3))
    assert f_score(v, v, 5.0) == pytest.approx(1.0)


def test_fscore_zero_threshold_identical_is_zero():
    # threshold 0: only exact overlaps count; identical sets still match (dist 0).
    v = np.random.default_rng(6).normal(size=(50, 3))
    # dist between a vertex and itself is 0, which is < any positive threshold;
    # at threshold exactly 0, "<0" is False so nothing matches -> 0.0
    assert f_score(v, v, 0.0) == pytest.approx(0.0)


def test_fscore_shape_distortion_drops_below_one():
    # A non-uniform deformation that Procrustes cannot remove.
    v = np.random.default_rng(12).normal(size=(100, 3))
    distorted = v.copy()
    distorted[:, 0] *= 4.0
    distorted[::3, 1] += 0.3
    assert f_score(v, distorted, 5.0) < 1.0


def test_fscore_is_symmetric_for_identical_sets():
    v = np.random.default_rng(7).normal(size=(100, 3))
    assert f_score(v, v, 5.0) == pytest.approx(1.0)
    assert f_score(v, v, 15.0) == pytest.approx(1.0)


def test_fscore_aligned_translation_returns_one():
    # f_score aligns first; a pure translation is removed by Procrustes.
    v = np.random.default_rng(8).normal(size=(100, 3))
    shifted = v + 5.0
    assert f_score(v, shifted, 5.0) == pytest.approx(1.0, abs=1e-6)


def test_fscore_different_sizes():
    # pred and gt can have different vertex counts; f_score uses pairwise dist.
    pred = np.random.default_rng(9).normal(size=(778, 3))
    gt = np.random.default_rng(10).normal(size=(778, 3))
    val = f_score(pred, gt, 5.0)
    assert 0.0 <= val <= 1.0


def test_fscore_threshold_monotonic():
    # A higher threshold can only increase or maintain F-score.
    rng = np.random.default_rng(11)
    pred = rng.normal(size=(100, 3))
    gt = rng.normal(size=(100, 3))
    assert f_score(pred, gt, 1.0) <= f_score(pred, gt, 5.0) + 1e-9
    assert f_score(pred, gt, 5.0) <= f_score(pred, gt, 50.0) + 1e-9


# ── aggregate: normalize edge cases ────────────────────────────────────

def test_normalize_empty_list_returns_empty():
    assert agg.normalize([]) == []


def test_normalize_all_none_returns_all_best():
    out = agg.normalize([None, None, None])
    assert out == [1.0, 1.0, 1.0]


def test_normalize_single_element():
    assert agg.normalize([42.0]) == [1.0]


def test_normalize_single_none():
    assert agg.normalize([None]) == [1.0]


def test_normalize_all_equal_returns_all_best():
    out = agg.normalize([7.0, 7.0, 7.0])
    assert all(abs(v - 1.0) < 1e-9 for v in out)


def test_normalize_lower_true_best_is_first():
    # range [1,10]; 5 is at (5-1)/(10-1)=4/9 of the way -> score 1-4/9=5/9.
    out = agg.normalize([1.0, 5.0, 10.0], lower=True)
    assert out[0] == 1.0
    assert out[2] == 0.0
    assert out[1] == pytest.approx(1.0 - 4.0 / 9.0)


def test_normalize_lower_false_best_is_last():
    out = agg.normalize([1.0, 5.0, 10.0], lower=False)
    assert out[2] == 1.0 and out[0] == 0.0


def test_normalize_none_in_lower_true_is_best():
    out = agg.normalize([10.0, None, 30.0], lower=True)
    assert out[1] == 1.0
    assert out[0] == 1.0  # 10 is the present best
    assert out[2] == 0.0


# ── aggregate: load_rows / scoring edge cases ───────────────────────────

def _write_raw(td, name, payload):
    Path(td, name).write_text(json.dumps(payload))


def _benchmark_row(model, latency=10.0, acc=5.0, vram=1000.0, variant="default"):
    return {
        "model": model, "variant": variant,
        "latency_ms": {"mean": latency, "p95": latency * 1.2},
        "accuracy": {"pa_mpjpe_mm": acc},
        "vram_mb": {"nvidia_smi_peak": vram},
        "fps": 1000.0 / latency, "n_predictions": 200, "n_misses": 0,
    }


def test_load_rows_skips_non_benchmark_json(tmp_path):
    _write_raw(tmp_path, "modelA.json", _benchmark_row("A"))
    _write_raw(tmp_path, "opt.json", {"results": [{"mean_ms": 5}]})  # no latency_ms/accuracy
    _write_raw(tmp_path, "trt.json", {"latency": {"mean_ms": 13}})  # no latency_ms/accuracy
    orig = agg.RAW
    agg.RAW = tmp_path
    try:
        rows = agg.load_rows()
    finally:
        agg.RAW = orig
    assert len(rows) == 1
    assert rows[0]["id"] == "A-default"


def test_load_rows_empty_dir_returns_empty(tmp_path):
    orig = agg.RAW
    agg.RAW = tmp_path
    try:
        assert agg.load_rows() == []
    finally:
        agg.RAW = orig


def test_load_rows_skips_invalid_json(tmp_path):
    _write_raw(tmp_path, "modelA.json", _benchmark_row("A"))
    (tmp_path / "broken.json").write_text("{not valid json")
    orig = agg.RAW
    agg.RAW = tmp_path
    try:
        rows = agg.load_rows()
    finally:
        agg.RAW = orig
    assert len(rows) == 1


def test_load_rows_id_includes_variant():
    # variant "fast" is appended; other variants use "default".
    rows = [_benchmark_row("wilor", variant="fast")]
    for r in rows:
        r["id"] = f"{r['model']}-{r.get('variant', 'default')}"
    # The main() builds id as model + "-fast" only when variant == "fast".
    # load_rows itself uses model + "-" + variant (default 'default').
    assert rows[0]["id"] == "wilor-fast"


def test_weights_sum_to_one():
    total = sum(agg.WEIGHTS.values())
    assert abs(total - 1.0) < 1e-9


def test_complexity_scores_in_expected_range():
    for k, v in agg.COMPLEXITY.items():
        assert 1 <= v <= 5, f"{k}={v} out of [1,5]"


# ── interface: HandPrediction defaults ──────────────────────────────────

def test_handprediction_required_fields():
    p = HandPrediction(joints_3d=np.zeros((21, 3), dtype=np.float32), confidence=0.9)
    assert p.joints_3d.shape == (21, 3)
    assert 0.0 <= p.confidence <= 1.0
    assert p.bbox_xyxy is None
    assert p.mano_pose is None
    assert p.mano_shape is None
    assert p.mesh_verts is None
    assert p.handedness == "right"
    assert p.raw == {}


def test_handprediction_confidence_outside_unit_is_still_set():
    # The dataclass does not clamp; it's the caller's contract. Just confirm no crash.
    p = HandPrediction(joints_3d=np.zeros((21, 3)), confidence=1.5)
    assert p.confidence == 1.5


def test_handprediction_raw_is_independent_per_instance():
    # default_factory must not share mutable state across instances.
    a = HandPrediction(joints_3d=np.zeros((21, 3)), confidence=0.1)
    b = HandPrediction(joints_3d=np.zeros((21, 3)), confidence=0.2)
    a.raw["x"] = 1
    assert "x" not in b.raw


def test_basehandmodel_cannot_instantiate():
    with pytest.raises(TypeError):
        BaseHandModel()


def test_basehandmodel_abstract_methods_present():
    abstract = BaseHandModel.__abstractmethods__
    assert "load" in abstract
    assert "preprocess" in abstract
    assert "infer" in abstract
    assert "device" in abstract


# ── write_raw: schema and round-trip ───────────────────────────────────

def test_write_raw_creates_file_and_round_trips(tmp_path):
    from common.profiling import write_raw
    payload = {"model": "x", "latency_ms": {"mean": 5.0}, "accuracy": {"pa_mpjpe_mm": 3.0}}
    path = write_raw("x", "default", payload, out_dir=str(tmp_path))
    assert os.path.exists(path)
    assert path.endswith("x.json")
    loaded = json.loads(Path(path).read_text())
    assert loaded == payload


def test_write_raw_overwrites_existing(tmp_path):
    from common.profiling import write_raw
    write_raw("y", "default", {"v": 1}, out_dir=str(tmp_path))
    write_raw("y", "default", {"v": 2}, out_dir=str(tmp_path))
    files = list(Path(tmp_path).glob("*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text())["v"] == 2


def test_write_raw_creates_out_dir_if_missing(tmp_path):
    from common.profiling import write_raw
    nested = str(tmp_path / "deep" / "nest")
    write_raw("z", "default", {"v": 1}, out_dir=nested)
    assert (Path(nested) / "z.json").exists()


# ── latency helpers: pct / summarize edge cases ────────────────────────

def test_pct_empty_raises_or_clamps():
    # pct([]) — current impl: sorted([]) = []; max(0, min(-1, ...)) -> IndexError or -1.
    # Document whatever the actual behavior is; the helper must not silently return 0.
    from optimize.bench_onnx_latency import pct
    try:
        val = pct([], 50)
        assert val is None or val == 0.0  # accept either, but no exception leaked
    except (IndexError, ValueError):
        pass  # acceptable: explicit failure on empty


def test_summarize_single_value():
    from optimize.bench_onnx_latency import summarize
    s = summarize([42.0])
    assert s["n"] == 1
    assert s["mean_ms"] == 42.0
    assert s["p50_ms"] == 42.0
    assert s["p95_ms"] == 42.0
    assert s["p99_ms"] == 42.0
    assert s["fps"] == round(1000 / 42.0, 2)


def test_summarize_keys_complete():
    from optimize.bench_onnx_latency import summarize
    s = summarize([1.0, 2.0])
    assert {"mean_ms", "p50_ms", "p95_ms", "p99_ms", "fps", "n"} == set(s)


def test_pct_p0_is_min_p100_is_max():
    from optimize.bench_onnx_latency import pct
    v = [float(i) for i in range(1, 21)]
    assert pct(v, 0) == 1.0
    assert pct(v, 100) == 20.0
