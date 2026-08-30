"""White-box property tests for data/freihand/loader.py.

These are marked to skip if FREIHAND_ROOT is unset (no data in the env), so they
run only where the real dataset is present (WSL egohand env). They assert the
indexing round-trip and subset-containment invariants the benchmark depends on.
Run: python -m pytest tests/test_loader_whitebox.py -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.freihand.loader import FreiHandLoader  # noqa: E402

FREIHAND = os.environ.get("FREIHAND_ROOT")
pytestmark = pytest.mark.skipif(not FREIHAND, reason="FREIHAND_ROOT unset; needs real data")


def test_indices_bounded_and_unique():
    L = FreiHandLoader(subset="data/freihand/subsets/dev.json")
    assert len(L) == 200
    assert len(set(L.indices)) == len(L.indices)
    assert all(0 <= i < L.n for i in L.indices)


def test_getitem_returns_indexed_gt():
    # __getitem__(k) must return GT for indices[k], not raw k. Verify by
    # cross-checking the loader's internal joints array at indices[k].
    L = FreiHandLoader()
    k = 3
    img, j, K = L[k]
    np.testing.assert_allclose(j, L.joints[L.indices[k]])


def test_get_gt_verts_consistency():
    L = FreiHandLoader()
    v0 = L.get_gt_verts(5)
    v1 = L.get_gt_verts(5)
    np.testing.assert_array_equal(v0, v1)


def test_dev_subset_is_subset_of_full():
    dev = set(FreiHandLoader(subset="data/freihand/subsets/dev.json").indices)
    full = set(FreiHandLoader(subset="data/freihand/subsets/full.json").indices)
    assert dev.issubset(full)


def test_image_filename_matches_indices():
    L = FreiHandLoader(subset="data/freihand/subsets/dev.json")
    for k in (0, 50, 199):
        stem = int(L.image_path(k).stem)
        assert stem == L.indices[k]
