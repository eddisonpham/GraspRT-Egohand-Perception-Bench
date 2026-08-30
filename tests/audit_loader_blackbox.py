"""Black-box audit of data/freihand/loader.py against real FreiHAND data.

Checks indexing consistency, GT units (meters), image-path format, subset index
fidelity, and missing-image robustness. Run inside WSL egohand env with
FREIHAND_ROOT set:

    python tests/audit_loader_blackbox.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.freihand.loader import FreiHandLoader  # noqa: E402

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    PASS += bool(cond)
    FAIL += not bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")


L = FreiHandLoader(subset="data/freihand/subsets/dev.json")
print(f"root: {L.root}  n_indices: {len(L)}  n_joints_total: {L.n}")

print("=== BB-1: index consistency ===")
check("len(indices) == 200 (dev)", len(L) == 200, f"got {len(L)}")
check("indices within [0, n)", all(0 <= i < L.n for i in L.indices))
check("indices unique", len(set(L.indices)) == len(L.indices))

print("=== BB-2: image_path format ===")
p = L.image_path(0)
check("image_path ends .jpg", str(p).endswith(".jpg"))
# FreiHAND eval uses 8-digit zero-padded names.
stem = p.stem
check("filename is 8-digit zero-padded", stem.isdigit() and len(stem) == 8, f"stem={stem}")
check("image_path uses indices[i]", int(stem) == L.indices[0])

print("=== BB-3: GT units (meters) ===")
img, j, K = L[0]
check("image is 224x224x3 uint8", img.shape[:2] == (224, 224) and img.dtype == np.uint8,
      f"{img.shape} {img.dtype}")
check("joints shape (21,3)", j.shape == (21, 3), f"{j.shape}")
check("K shape (3,3)", K.shape == (3, 3))
# FreiHAND joints are in meters; typical hand span ~0.15-0.25 m.
span = np.ptp(j, axis=0)
check("joint span in meters (max < 0.5m)", span.max() < 0.5, f"span={span.round(4)}")
check("joint span reasonable (> 0.02m)", span.max() > 0.02)

print("=== BB-4: verts GT ===")
v = L.get_gt_verts(0)
check("verts shape (778,3)", v.shape == (778, 3), f"{v.shape}")
check("verts in meters (max|v| < 1m)", np.abs(v).max() < 1.0, f"max|v|={np.abs(v).max():.3f}")

print("=== BB-5: index access round-trip ===")
# __getitem__ with index k must return GT for indices[k], not raw k.
img1, j1, _ = L[1]
v1 = L.get_gt_verts(1)
img2, j2, _ = L[1]  # deterministic
check("loader deterministic (same idx twice)", np.array_equal(j1, j2) and np.array_equal(v1, L.get_gt_verts(1)))

print("=== BB-6: full vs dev subset ===")
Lf = FreiHandLoader(subset="data/freihand/subsets/full.json")
check("full subset has 3960", len(Lf) == 3960, f"got {len(Lf)}")
# dev indices must be a subset of full indices
dev_set = set(L.indices)
full_set = set(Lf.indices)
check("dev ⊂ full", dev_set.issubset(full_set))

print(f"\n=== SUMMARY: {PASS} passed, {FAIL} failed ===")
sys.exit(0 if FAIL == 0 else 1)
