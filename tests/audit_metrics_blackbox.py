"""Black-box audit of common/metrics.py (no GPU/model, pure NumPy).

Runs each metric on inputs with KNOWN expected outputs and asserts the measured
value matches a documented tolerance. Each case prints the actual number so a
human can eyeball validity. Run:

    python tests/audit_metrics_blackbox.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.metrics import f_score, mpjpe, pa_mpjpe, pa_mpvpe, procrustes_align  # noqa: E402

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    flag = "PASS" if cond else "FAIL"
    if cond:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{flag}] {name}  {detail}")


def fmt(v):
    return f"{v:.6g}"


rng = np.random.default_rng(42)

print("=== BB-1: procrustes_align invariants ===")
gt = rng.normal(size=(21, 3))

# (a) Identity: align(gt, gt) == gt exactly.
aligned = procrustes_align(gt, gt)
err = np.abs(aligned - gt).max()
check("identity align == gt", err < 1e-9, f"max|d|={err:.2e}")

# (b) Similarity recovery: a known R/scale/t applied to gt aligns back to ~0 PA-MPJPE.
R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
scale, t = 2.5, np.array([0.1, -0.2, 0.3])
pred = (gt @ R.T) * scale + t
e = pa_mpjpe(pred, gt)
check("similarity-transform PA-MPJPE ~ 0", e < 1e-6, f"PA-MPJPE={fmt(e)}mm")

# (c) Reflection is disallowed: mirror (det=-1) must not be solved by reflecting.
#   Apply an improper rotation (reflection); PA should still be small only via the
#   nearest proper rotation, leaving a residual we can bound.
M = np.diag([-1.0, 1.0, 1.0])  # reflection across yz-plane, det=-1
pred_refl = (gt @ M.T) * 1.0
e_refl = pa_mpjpe(pred_refl, gt)
# The best *proper* rotation mapping gt->reflected gt is identity with a flip in x
# that proper rotation cannot represent exactly -> residual > 0.
check("reflection not used (residual > 0)", e_refl > 1e-3, f"PA-MPJPE={fmt(e_refl)}mm")

print("=== BB-2: pa_mpjpe / pa_mpvpe unit checks ===")
# Single-joint 1mm perturbation is partly absorbed by the global similarity fit
# (scale/rotation/translation are fit to ALL joints), so the naive 1/21mm
# expectation is wrong. Verify instead: (a) it's positive and bounded, and
# (b) the perturbed joint carries the largest residual.
gt21 = rng.normal(size=(21, 3))
shift = gt21.copy()
shift[0] += np.array([0.001, 0.0, 0.0])  # 1 mm in meters
from common.metrics import procrustes_align
per = np.linalg.norm(procrustes_align(shift, gt21) - gt21, axis=-1) * 1000.0
e = per.mean()
check("1mm single-joint: 0 < PA-MPJPE < 1mm", 0 < e < 1.0, f"PA-MPJPE={fmt(e)}mm")
check("perturbed joint has the largest residual", np.argmax(per) == 0,
      f"joint0={fmt(per[0])}mm vs max={fmt(per.max())}mm")

# 778 verts, uniform 5mm shift on all -> PA removes it (translation) -> ~0.
v778 = rng.normal(size=(778, 3))
shifted_v = v778 + np.array([0.005, 0.0, 0.0])
e_v = pa_mpvpe(shifted_v, v778)
check("uniform translation removed by PA (PA-MPVPE~0)", e_v < 1e-6, f"PA-MPVPE={fmt(e_v)}mm")

print("=== BB-3: mpjpe (non-aligned) unit ===")
gt3 = np.zeros((3, 3))
pred3 = np.array([[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]])  # 1mm each
e = mpjpe(pred3, gt3)
check("3 joints each 1mm -> mpjpe=1mm", abs(e - 1.0) < 1e-6, f"mpjpe={fmt(e)}mm")

print("=== BB-4: f_score boundary cases ===")
v = rng.normal(size=(100, 3))
# (a) identical -> 1.0 at any threshold > 0
check("f_score(v,v,5mm)==1", abs(f_score(v, v, 5.0) - 1.0) < 1e-9)
# (b) pure translation offset -> F=1, because f_score aligns first (by design,
#     matching the PA-MPVPE protocol). A pure translation is similarity-removable.
off = v + 0.1
check("pure translation offset: F@5mm==1 (PA removes it)",
      abs(f_score(v, off, 5.0) - 1.0) < 1e-9)
# (c) per-vertex jitter (NOT similarity-removable): 6mm iid jitter per vertex.
#     F@5mm must drop below 1; F@20mm should be ~1.
jitter = v + rng.normal(scale=0.006, size=v.shape)  # 6mm per-vertex
f5 = f_score(v, jitter, 5.0)
f20 = f_score(v, jitter, 20.0)
check("6mm per-vertex jitter: F@5mm < 1", f5 < 1.0, f"F@5={fmt(f5)}")
check("6mm per-vertex jitter: F@20mm close to 1", f20 > 0.95, f"F@20={fmt(f20)}")

print("=== BB-5: shape/dtype robustness ===")
p32 = rng.normal(size=(21, 3)).astype(np.float32)
g64 = rng.normal(size=(21, 3))
a = procrustes_align(p32, g64)
check("mixed f32/f64 -> f64 aligned", a.dtype == np.float64 and a.shape == (21, 3))
try:
    procrustes_align(np.zeros((20, 3)), np.zeros((21, 3)))
    check("shape mismatch raises", False)
except ValueError:
    check("shape mismatch raises", True)

print(f"\n=== SUMMARY: {PASS} passed, {FAIL} failed ===")
sys.exit(0 if FAIL == 0 else 1)
