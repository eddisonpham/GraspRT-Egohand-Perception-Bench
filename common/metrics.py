"""Shared evaluation metrics (pure NumPy, all errors in millimeters).

Procrustes alignment is a similarity transform (rotation + translation +
uniform scale, no reflection), per FreiHAND PA-MPJPE/PA-MPVPE.
"""
from __future__ import annotations

import numpy as np


def _umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Fit scale, R, t mapping src onto dst: src @ R.T * scale + t ~= dst."""
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
        s[2, 2] = -1
    r = u @ s @ vt
    if with_scale:
        var_src = np.sum(src_c ** 2) / n
        scale = np.trace(np.diag(d) @ s) / var_src if var_src > 0 else 1.0
    else:
        scale = 1.0
    t = mu_dst - scale * (r @ mu_src)
    return scale, r, t


def procrustes_align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Align pred (N,3) onto gt (N,3) with a similarity transform."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    scale, r, t = _umeyama(pred, gt, with_scale=True)
    return (pred @ r.T) * scale + t


def mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean per-joint position error in mm, without alignment."""
    d = np.asarray(pred, np.float64) - np.asarray(gt, np.float64)
    return float(np.linalg.norm(d, axis=-1).mean() * 1000.0)


def pa_mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    """Procrustes-aligned mean per-joint position error in mm."""
    aligned = procrustes_align(pred, gt)
    return float(np.linalg.norm(aligned - np.asarray(gt, np.float64), axis=-1).mean() * 1000.0)


def pa_mpvpe(pred_verts: np.ndarray, gt_verts: np.ndarray) -> float:
    """Procrustes-aligned mean per-vertex position error in mm."""
    aligned = procrustes_align(pred_verts, gt_verts)
    return float(np.linalg.norm(aligned - np.asarray(gt_verts, np.float64), axis=-1).mean() * 1000.0)


def f_score(pred_verts: np.ndarray, gt_verts: np.ndarray, threshold_mm: float) -> float:
    """F-score at a distance threshold in mm (aligns first, like PA-MPVPE)."""
    pred = procrustes_align(pred_verts, gt_verts) * 1000.0
    gt = np.asarray(gt_verts, np.float64) * 1000.0
    dist = np.linalg.norm(pred[:, None, :] - gt[None, :, :], axis=-1)
    prec = (dist.min(axis=1) < threshold_mm).mean()
    rec = (dist.min(axis=0) < threshold_mm).mean()
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))