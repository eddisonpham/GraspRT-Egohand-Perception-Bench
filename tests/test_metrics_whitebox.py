"""White-box property tests for common/metrics.py — formal invariants.

These target the *implementation* of _umeyama/procrustes_align directly (symmetry,
reflection rejection, scale recovery) rather than end-to-end known answers.
Run: python -m pytest tests/test_metrics_whitebox.py -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.metrics import _umeyama, procrustes_align, pa_mpjpe  # noqa: E402


def test_umeyama_recovers_exact_transform():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(50, 3))
    R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    s, t = 3.1, np.array([0.5, -0.4, 0.2])
    dst = (src @ R.T) * s + t
    s_hat, R_hat, t_hat = _umeyama(src, dst, with_scale=True)
    # R_hat must be a proper rotation (det = +1).
    assert np.isclose(np.linalg.det(R_hat), 1.0, atol=1e-9), f"det(R)={np.linalg.det(R_hat)}"
    assert np.allclose(s_hat, s, atol=1e-6), f"scale {s_hat} vs {s}"
    # Round-trip: dst ≈ src @ R_hat.T * s_hat + t_hat
    recon = (src @ R_hat.T) * s_hat + t_hat
    assert np.allclose(recon, dst, atol=1e-6)


def test_umeyama_no_scale_when_requested():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(30, 3))
    s_hat, _, _ = _umeyama(src, src * 2.0, with_scale=False)
    assert np.isclose(s_hat, 1.0, atol=1e-9)


def test_procrustes_rejects_reflection():
    rng = np.random.default_rng(2)
    src = rng.normal(size=(40, 3))
    M = np.diag([-1.0, 1.0, 1.0])  # reflection, det=-1
    aligned = procrustes_align(src @ M.T, src)
    # Residual must be large: a proper rotation cannot perfectly undo a reflection.
    assert np.linalg.norm(aligned - src, axis=-1).mean() > 1e-3


def test_procrustes_translation_invariance():
    """Shifting both pred and gt by the same vector must not change PA-MPJPE."""
    rng = np.random.default_rng(3)
    gt = rng.normal(size=(21, 3))
    pred = gt + rng.normal(scale=0.01, size=gt.shape)
    base = pa_mpjpe(pred, gt)
    shifted = pa_mpjpe(pred + 5.0, gt + 5.0)
    assert np.isclose(base, shifted, atol=1e-6), f"{base} vs {shifted}"


def test_procrustes_scale_invariance():
    """Uniform scaling of pred only must give ~0 PA-MPJPE (scale is fit)."""
    rng = np.random.default_rng(4)
    gt = rng.normal(size=(21, 3))
    pred = gt * 7.3 + np.array([1.0, -2.0, 0.5])
    assert pa_mpjpe(pred, gt) < 1e-6


def test_umeyama_degenerate_input():
    """All-identical points: covariance is zero; must not NaN."""
    pts = np.ones((10, 3))
    s, R, t = _umeyama(pts, pts, with_scale=True)
    assert np.isfinite(s) and np.isfinite(R).all() and np.isfinite(t).all()
