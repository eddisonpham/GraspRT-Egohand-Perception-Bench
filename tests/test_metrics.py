"""Unit tests for common.metrics — run with: python -m pytest tests/test_metrics.py -q"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.metrics import (  # noqa: E402
    f_score,
    mpjpe,
    pa_mpjpe,
    pa_mpvpe,
    procrustes_align,
)


def test_pa_mpjpe_identity():
    gt = np.random.default_rng(0).normal(size=(21, 3))
    assert pa_mpjpe(gt, gt) == pytest.approx(0.0, abs=1e-9)


def test_pa_mpvpe_identity():
    gt = np.random.default_rng(1).normal(size=(778, 3))
    assert pa_mpvpe(gt, gt) == pytest.approx(0.0, abs=1e-9)


def test_similarity_invariance():
    gt = np.random.default_rng(2).normal(size=(21, 3))
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pred = (gt @ R.T) * 1.7 + np.array([0.3, -0.1, 0.2])
    assert pa_mpjpe(pred, gt) == pytest.approx(0.0, abs=1e-6)


def test_mpjpe_not_invariant():
    gt = np.random.default_rng(3).normal(size=(21, 3))
    pred = gt + 0.05
    assert mpjpe(pred, gt) == pytest.approx(0.05 * np.sqrt(3) * 1000.0, rel=0.01)  # 5 cm in xyz


def test_fscore_perfect_overlap():
    v = np.random.default_rng(4).normal(size=(100, 3))
    assert f_score(v, v, 5.0) == pytest.approx(1.0)


def test_fscore_penalizes_non_similarity_shape_error():
    v = np.random.default_rng(5).normal(size=(100, 3))
    distorted = v.copy()
    # A non-uniform, vertex-dependent deformation cannot be removed by Procrustes.
    distorted[:, 0] *= 4.0
    distorted[::2, 1] += 0.5
    assert f_score(v, distorted, 5.0) < 1.0


def test_procrustes_shape_and_dtype():
    pred = np.random.default_rng(6).normal(size=(21, 3)).astype(np.float32)
    gt = np.random.default_rng(7).normal(size=(21, 3))
    aligned = procrustes_align(pred, gt)
    assert aligned.shape == (21, 3)
    assert aligned.dtype == np.float64