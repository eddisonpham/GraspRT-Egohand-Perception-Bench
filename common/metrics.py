"""Evaluation metrics shared by every benchmark run.

All alignment/error code is pure NumPy so it runs identically in every model env.

Procrustes alignment here means a *similarity transform* (rotation + translation +
uniform scale, NO reflection) — standard for FreiHAND PA-MPJPE / PA-MPVPE.
Implemented as Umeyama (1991). All errors are reported in millimeters.
"""
from __future__ import annotations

import numpy as np


def _umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Similarity transform mapping src -> dst.

    Returns (scale, R, t) such that src @ R.T * scale + t ≈ dst.
    src, dst: (N, 3) arrays.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    n = src.shape[0]
    cov = (dst_c.T @ src_c) / n
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[2, 2] = -1  # disallow reflection
    r = u @ s @ vt
    if with_scale:
        var_src = np.sum(src_c ** 2) / n
        scale = np.trace(np.diag(d) @ s) / var_src if var_src > 0 else 1.0
    else:
        scale = 1.0
    t = mu_dst - scale * (r @ mu_src)
    return scale, r, t


def procrustes_align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Similarity-transform-align pred (N,3) onto gt (N,3). Returns aligned pred."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    scale, r, t = _umeyama(pred, gt, with_scale=True)
    return (pred @ r.T) * scale + t


def mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean per-joint position error in mm, no alignment."""
    return float(np.linalg.norm(np.asarray(pred, np.float64) - np.asarray(gt, np.float64), axis=-1).mean() * 1000.0)


def pa_mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    """Procrustes-Aligned MPJPE in mm. THE headline accuracy metric for this project."""
    aligned = procrustes_align(pred, gt)
    return float(np.linalg.norm(aligned - np.asarray(gt, np.float64), axis=-1).mean() * 1000.0)


def pa_mpvpe(pred_verts: np.ndarray, gt_verts: np.ndarray) -> float:
    """Procrustes-Aligned mean per-vertex position error in mm. Only for models with mesh_verts."""
    aligned = procrustes_align(pred_verts, gt_verts)
    return float(np.linalg.norm(aligned - np.asarray(gt_verts, np.float64), axis=-1).mean() * 1000.0)


def f_score(pred_verts: np.ndarray, gt_verts: np.ndarray, threshold_mm: float) -> float:
    """F-score at a distance threshold (5mm and 15mm are the standard FreiHAND thresholds)."""
    # Align first (same protocol as PA-MPVPE), then compare in mm. The previous
    # implementation compared aligned mm predictions against unaligned meters GT,
    # producing the invalid all-zero F-scores caught by WiLoR's first run.
    pred = procrustes_align(pred_verts, gt_verts) * 1000.0  # meters -> mm
    gt = np.asarray(gt_verts, np.float64) * 1000.0
    dist = np.linalg.norm(pred[:, None, :] - gt[None, :, :], axis=-1)  # (P, G)
    prec = (dist.min(axis=1) < threshold_mm).mean()  # how many pred verts have a GT vert nearby
    rec = (dist.min(axis=0) < threshold_mm).mean()   # how many GT verts have a pred vert nearby
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    gt = rng.normal(size=(21, 3))
    # 1. perfect self-alignment
    assert abs(pa_mpjpe(gt, gt)) < 1e-9, "pa_mpjpe(gt, gt) must be 0"
    assert abs(pa_mpvpe(gt, gt)) < 1e-9, "pa_mpvpe(gt, gt) must be 0"
    # 2. similarity-transformed pred must align back to ~0
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pred = (gt @ R.T) * 2.0 + np.array([0.1, -0.2, 0.3])
    assert abs(pa_mpjpe(pred, gt)) < 1e-6, f"similarity-invariant PA-MPJPE broken: {pa_mpjpe(pred, gt)}"
    # 3. mpjpe is NOT similarity-invariant (same as PA only when no transform applied)
    assert abs(mpjpe(gt, gt)) < 1e-9
    assert mpjpe(pred, gt) > pa_mpjpe(pred, gt)
    # 4. f-score: identical sets -> 1.0 at any threshold above 0
    assert abs(f_score(gt, gt, 5.0) - 1.0) < 1e-9
    flipped = gt + np.array([1.0, 0.0, 0.0])  # 1 m shift in mm-space... gt is in meters here
    # gt in meters: shift by 1e-4 m = 0.1 mm -> F@5mm stays 1, F@0.05mm drops
    shifted = gt + 1e-4
    assert abs(f_score(gt, shifted, 5.0) - 1.0) < 1e-9
    assert f_score(gt, gt + 0.1, 5.0) < 1.0
    print("metrics self-test: all assertions passed")