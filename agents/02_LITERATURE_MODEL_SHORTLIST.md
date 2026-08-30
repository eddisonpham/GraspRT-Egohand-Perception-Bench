# 02 — Literature Review & Model Shortlist

Four candidates, deliberately spanning the accuracy/speed/model-size Pareto frontier rather than
four similar models. Each entry: what it is, why it's here, its VRAM/speed profile going in, and
its licensing/gate caveats.

---

### Candidate A — MediaPipe Hand Landmarker (floor reference, no mesh)

- **What:** Google's two-stage palm-detector + 21-keypoint landmark model. Outputs 2D image-space
  landmarks, 3D "world" landmarks, and handedness. No mesh, no MANO parameters.
- **Repo:** https://github.com/google-ai-edge/mediapipe
- **Model asset:** `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`
- **Docs:** https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
- **Why it's here:** Inductive bias #4 (see `00`) — you need a deliberately cheap floor reference.
  Real-time on CPU alone; if a heavier model can't clearly beat this on the metrics that matter for
  your use case, that's a real finding, not a failure.
- **Expected profile:** Tiny (a few MB), CPU-real-time, lowest expected accuracy on 3D metrics
  since it's optimized for 2D/AR use cases, not metric-accurate 3D reconstruction.
- **License/gate:** Apache 2.0, no registration needed. Easiest to bring up — do this one first to
  validate the harness before touching the heavier three.

### Candidate B — MobRecon (lightweight MANO mesh)

- **Paper:** Chen et al., *"MobRecon: Mobile-Friendly Hand Mesh Reconstruction from Monocular
  Image"*, CVPR 2022. arXiv: https://arxiv.org/abs/2112.02753
- **Repo:** https://github.com/SeanChenxy/HandMesh
- **Why it's here:** Purpose-built for exactly this constraint class (the paper reports 83 FPS on
  an *Apple A14 CPU*). Uses a lightweight 2D stacked-hourglass-style encoder plus an efficient
  graph-conv ("depth-separable spiral convolution") decoder to regress a MANO mesh — this is the
  "someone already solved your problem, go verify it" candidate.
- **Expected profile:** Small parameter count, should comfortably fit in your VRAM budget with
  large headroom; accuracy benchmarked by the authors on FreiHAND, RHD, HO3Dv2.
- **License/gate:** Code is open; **requires a MANO model download**, which requires a one-time,
  manual, click-through registration at https://mano.is.tue.mpg.de/ — this cannot be automated
  (see `05` and `12`).

### Candidate C — WiLoR (real-time transformer, ships a fast mode)

- **Paper:** Potamias et al., *"WiLoR: End-to-end 3D Hand Localization and Reconstruction
  in-the-wild"*, CVPR 2025. arXiv: https://arxiv.org/abs/2409.12259
- **Repo:** https://github.com/rolpotamias/WiLoR (project page:
  https://rolpotamias.github.io/WiLoR)
- **Why it's here:** Two-stage pipeline — a real-time fully-convolutional hand *detector* (trained
  on a 2M-image in-the-wild dataset the authors released) feeding a ViT-based MANO reconstruction
  head. Directly matches inductive bias #1 (decoupled detector). It **ships a built-in `--fast`
  flag** (FP16 + depth pruning) claiming up to 1.6x speedup for ~0.05mm accuracy cost — bias #2.
  It is also the paper's own stated comparison point against HaMeR, positioning it as the
  efficiency-focused alternative to the accuracy-focused Candidate D.
- **Expected profile:** Real-time-oriented by design; ViT backbone means non-trivial VRAM vs
  MobRecon/MediaPipe, but explicitly built for practical deployment, not just benchmark accuracy.
- **License/gate:** WiLoR model weights are **CC-BY-NC-ND** (non-commercial, no-derivatives) —
  fine for this portfolio/research project, note it in any writeup. Also requires the MANO
  registration above. Depends on `ultralytics` for its detector stage.

### Candidate D — HaMeR (accuracy ceiling / reference point, expected to be heavy)

- **Paper:** Pavlakos et al., *"Reconstructing Hands in 3D with Transformers"*, CVPR 2024. arXiv:
  https://arxiv.org/abs/2312.05251
- **Repo:** https://github.com/geopavlakos/hamer
- **Why it's here:** Fully transformer-based (ViT-H backbone), the accuracy state-of-the-art that
  WiLoR itself is benchmarked against, and placed **2nd in the Ego-Pose Hands task of the
  Ego-Exo4D Challenge** — directly relevant to the egocentric domain this pipeline targets. It is
  included specifically as an **accuracy ceiling / sanity check**, not as an expected winner: a
  ViT-H backbone is a large model, and part of the point of this benchmark is confirming, with real
  numbers, that it either does or doesn't fit your budget — don't assume either way.
- **Expected profile:** Highest expected accuracy, highest expected latency and VRAM of the four.
  If it doesn't fit even at batch size 1 / FP16, that's a valid, useful, and reportable result —
  run it anyway on a handful of images to get an accuracy number even if it's benchmarked at
  reduced iteration count.
- **License/gate:** Requires MANO registration (same gate as B and C).
- **Related work worth citing but not implementing:** *Fast-HaMeR: Boosting Hand Mesh
  Reconstruction using Knowledge Distillation* (2026) explores distilling HaMeR's ViT-H backbone
  into MobileNet/MobileViT/ConvNeXt/ResNet students, reporting ~1.5x speedup at ~35% of the
  original size for ~0.4mm accuracy cost. No public trained checkpoint is assumed to exist — this
  is cited in `07`/`REFERENCES.md` as forward-looking context for *why* a distillation-based
  optimization path is plausible if the ONNX/TensorRT route on Candidate D doesn't get you far
  enough, not as a 5th thing to implement now.

---

## Common evaluation ground

All four candidates will be evaluated on the same benchmark: **FreiHAND** (see `04`), using **3D
keypoint accuracy (PA-MPJPE)** as the one metric every candidate can produce (MediaPipe gives
keypoints directly; B/C/D derive keypoints from their MANO output). Mesh-level metrics
(PA-MPVPE, F-scores) are reported as a bonus for B/C/D since MediaPipe has no mesh to compare.

## Definition of Done

- [ ] You can name, for each of the 4 candidates, one concrete architectural reason it might win
      and one concrete reason it might lose, in your own words, in `results/02_notes.md`.
- [ ] You have located and can reach (not yet cloned) all 4 repos and the MediaPipe model asset URL.
- [ ] You understand that B, C, and D all share the MANO registration gate — flag this to the user
      now if it hasn't been done yet, since it's the one manual step you cannot do for them.
