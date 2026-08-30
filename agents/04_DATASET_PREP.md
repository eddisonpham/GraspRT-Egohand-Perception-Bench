# 04 — Dataset Preparation

Two datasets, two different jobs: **FreiHAND** gives you ground truth for quantitative accuracy
numbers; a **short egocentric clip you record yourself** gives you a qualitative, domain-relevant
sanity check (no ground truth, just "does this look right for the actual use case").

## Part 1 — FreiHAND (quantitative benchmark)

FreiHAND is directly downloadable, no login wall, research-only/non-commercial license (cite the
paper if this ever leaves the portfolio-project stage).

```bash
mkdir -p egohand-bench/data/freihand && cd egohand-bench/data/freihand
wget https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip
unzip -q FreiHAND_pub_v2.zip
```

This gives you `training/rgb/*.jpg` plus `training_xyz.json` (3D joint ground truth),
`training_K.json` (camera intrinsics), and similar files for the evaluation split. The evaluation
split's annotations have been made public directly on the dataset page (the original Codalab
server had ongoing issues) — check
https://lmb.informatik.uni-freiburg.de/projects/freihand/ for the current eval-annotation file
name/location if it's not bundled in the same zip, and download it alongside.

**Build two subsets, don't run everything every time:**

```python
# data/freihand/build_subsets.py
# - "dev" subset: 200 random eval-split images, for fast iteration while debugging wrappers
# - "full" subset: entire eval split (~3960 images), for the final numbers that go in the report
```

Write a small `FreiHandLoader` class (in `data/freihand/loader.py`) that:
1. Loads an image by index, returns `(image_bgr: np.ndarray, gt_joints_3d: np.ndarray (21,3), K: np.ndarray (3,3))`.
2. Optionally loads `gt_verts: np.ndarray (778,3)` for models that support mesh metrics.
3. Is used identically by every model wrapper's benchmark run — this is what keeps the comparison fair.

Note: MediaPipe's coordinate convention (world landmarks, hand-relative, meters, wrist-anchored)
and FreiHAND's convention (camera-space, meters) are **not the same**. Before computing PA-MPJPE,
Procrustes alignment (in `common/metrics.py`) removes the global similarity transform (rotation,
translation, scale) — this is *why* PA-MPJPE rather than raw MPJPE is the headline metric here: it
makes cross-convention comparison meaningful. Raw MPJPE is not directly comparable across models
that don't share a coordinate convention — do not report it as a primary number.

## Part 2 — Egocentric qualitative clips (domain sanity check, no ground truth)

Ego4D and EPIC-KITCHENS both require a manual, click-through license agreement to download — **not
automatable by a coding agent**, and not worth gating this project on. Instead:

1. Record 2–3 short (10–15 second) clips of your own hand manipulating an object, held or worn at
   a close, downward, first-person-ish angle — phone camera or laptop webcam is fine. This is a
   reasonable, honest proxy for "egocentric-style" input for a qualitative check; it does not need
   to be a research-grade dataset to be useful here.
2. Save them under `data/egocentric_clips/clip_01.mp4`, etc.
3. These are used in `06` and `11` purely for **visual side-by-side rollout comparison** across the
   4 models (does the predicted skeleton/mesh track the hand plausibly under egocentric-style
   motion blur, close range, and partial occlusion) — not for any quantitative score, since there's
   no ground truth. Say this explicitly in the final report so it isn't mistaken for a benchmark.

## Definition of Done

- [ ] `data/freihand/` contains the unzipped training + eval data and a working `FreiHandLoader`.
- [ ] `dev` (≈200 images) and `full` (full eval split) subset index files exist.
- [ ] You can load one sample and print its image shape, `gt_joints_3d` shape `(21,3)`, and `K`
      shape `(3,3)` with no errors.
- [ ] At least 2 short egocentric-style clips exist under `data/egocentric_clips/`.
- [ ] You have written, in one sentence in `results/04_notes.md`, why PA-MPJPE rather than raw
      MPJPE is the fair cross-model metric here.
