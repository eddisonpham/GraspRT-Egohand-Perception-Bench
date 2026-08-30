# 04 — Dataset Notes

## FreiHAND
- Downloaded **FreiHAND_pub_v2_eval.zip (724 MB)** — evaluation set WITH annotations (the
  Codalab-free public release): `evaluation/rgb/*.jpg` (3,960 imgs, 224x224), GT 3D joints
  `evaluation_xyz.json` (3960,21,3) [m], intrinsics `evaluation_K.json` (3960,3,3), GT mesh
  `evaluation_verts.json` (3960,778,3) [m], MANO params `evaluation_mano.json` + scale.
- `dev` subset: 200 seeded-random eval images → `data/freihand/subsets/dev.json`
- `full` subset: all 3,960 eval images → `data/freihand/subsets/full.json`
- `FreiHandLoader` verified: image (224,224,3) BGR, gt joints (21,3) [m], K (3,3).
- **Infra decision:** data lives primarily at `~/egohand_data/freihand/` (WSL ext4 — the 9p
  /mnt/c mount is far too slow for 3,960-file workloads). `FreiHandLoader` resolves
  `$FREIHAND_ROOT` → `~/egohand_data/freihand` → repo `data/freihand/`. So the repo tree stays
  clean while runs stay fast; the loader is the single source of truth for both.
- Tried downloading the full 3.7 GB `FreiHAND_pub_v2.zip` (training split) for stage-09 INT8
  calibration; the LMB server throttles badly. Re-attempt later; if it stays unusable, calibrate
  on held-out *eval* images (seeded, disjoint from the scored subset) and say so in 09 — the
  deviation is a small, documented one on a 200-image calibration set.

## Egocentric clips
- **Required from user** (cannot be recorded by the agent): 2-3 short (10-15 s) clips of a hand
  manipulating an object from a close, downward, first-person-ish angle → place under
  `data/egocentric_clips/clip_01.mp4` etc. Flagged to user in the 05 checkpoint.

## Why PA-MPJPE rather than raw MPJPE is the fair cross-model metric
MediaPipe outputs hand-relative world coordinates while FreiHAND ground truth is camera-space
meters — the global similarity transform between conventions can't be compared directly, so
Procrustes alignment (removing rotation/translation/scale) is what makes all 4 candidates
comparable on one number.