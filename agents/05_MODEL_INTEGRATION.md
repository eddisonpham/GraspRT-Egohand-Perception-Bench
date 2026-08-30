# 05 — Model Integration

Bring up the 4 candidates **in order of increasing risk**: MediaPipe → MobRecon → WiLoR → HaMeR.
This validates the harness on the easy case first, so when something breaks on WiLoR/HaMeR you
know it's the model, not your benchmark code.

**Isolation strategy:** these repos have overlapping-but-different dependency pins (PyTorch
versions, detectron2/ultralytics variants, etc.). Rather than fighting one shared environment,
give each model its own conda env (`egohand-mediapipe`, `egohand-mobrecon`, `egohand-wilor`,
`egohand-hamer`), and have each wrapper's `load()` assume it's running inside its own env. The
benchmark harness in `06` invokes each model's benchmark run as a **separate subprocess** using
that model's env, writing results to a shared `results/raw/<model>.json` — this decouples envs
completely while keeping one unified results format.

**Reproducibility:** immediately after cloning each repo, record the commit:
```bash
git rev-parse HEAD >> ../../results/commit_hashes.json   # append per-model, keep it simple k/v
```
Anyone re-running this later (including future-you) needs to know exactly which commit produced
which numbers — these research repos change over time.

---

## A. MediaPipe Hand Landmarker

```bash
conda create -n egohand-mediapipe python=3.10 -y && conda activate egohand-mediapipe
pip install mediapipe opencv-python numpy
mkdir -p egohand-bench/models/assets
wget -O egohand-bench/models/assets/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

In `models/mediapipe_wrapper.py`, implement `MediaPipeHandModel(BaseHandModel)`:
- `load()`: construct `mp.tasks.vision.HandLandmarker` from `hand_landmarker.task`, running mode
  `IMAGE` (use `VIDEO` mode later if you benchmark on the egocentric clips frame-by-frame).
- `preprocess()`: wrap the BGR ndarray into an `mp.Image` (`SRGB` format, so convert BGR→RGB first).
- `infer()`: call `detect()`, map `hand_world_landmarks` (already 3D, meters) into `HandPrediction.joints_3d`. No mesh — leave `mesh_verts=None`.
- This model has no GPU path in the standard Python Tasks API — run it on CPU. That's fine and
  expected; it's the floor reference (see `00`, bias #4). Report its CPU latency, not a VRAM number.

## B. MobRecon

```bash
conda create -n egohand-mobrecon python=3.9 -y && conda activate egohand-mobrecon
git clone https://github.com/SeanChenxy/HandMesh.git
cd HandMesh && git rev-parse HEAD   # record this
# Follow the repo's own README for exact torch/torch-geometric pins — spiral conv depends on
# torch-geometric, which is version-sensitive. Install matching your CUDA toolkit from 01.
```

**MANO gate:** MobRecon needs the MANO right-hand model file. Registration is manual:
1. Go to https://mano.is.tue.mpg.de/, create an account, accept the license, download `MANO_RIGHT.pkl`.
2. Place it wherever `HandMesh`'s README specifies (check `mano/` or `template/` in that repo).
3. **This step cannot be automated by the agent** — if it hasn't been done, stop and ask the user
   to do it, then continue.

In `models/mobrecon_wrapper.py`, implement `MobReconHandModel(BaseHandModel)`:
- `load()`: load the pretrained checkpoint the repo provides (check their README/releases for the
  FreiHAND-trained checkpoint specifically, since that matches your eval set).
- `preprocess()`: match the repo's expected crop/resize/normalization exactly — do not improvise
  this; mismatched preprocessing silently tanks accuracy without any error.
- `infer()`: forward pass, extract MANO `pose`/`shape`, regress `joints_3d` and `mesh_verts` from
  the MANO layer the repo already includes.

## C. WiLoR

```bash
conda create -n egohand-wilor python=3.10 -y && conda activate egohand-wilor
git clone --recursive https://github.com/rolpotamias/WiLoR.git
cd WiLoR && git rev-parse HEAD   # record this
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128  # match 01
pip install -r requirements.txt
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt -P ./pretrained_models/
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt -P ./pretrained_models/
```

**MANO gate:** same as MobRecon — WiLoR also needs the MANO model file, same manual registration.

In `models/wilor_wrapper.py`, implement `WiLoRHandModel(BaseHandModel)`:
- `load()`: use the repo's `load_wilor()` helper with the downloaded checkpoint + config.
- `preprocess()`: reuse the repo's `ViTDetDataset` preprocessing path rather than reimplementing
  it — this repo's detector + crop logic is nontrivial (bounding box, rescale factor) and is not
  worth re-deriving.
- `infer()`: run detector → crop → reconstruction head, extract MANO output the same way as B.
- **License note:** weights are CC-BY-NC-ND — fine here, just don't repackage/redistribute them.
- Try the built-in fast mode too, as a *separate* config the benchmark harness treats as its own
  row (`wilor` vs `wilor-fast`) — this is one of your 4 core comparisons, and the `--fast` mode
  literally IS FP16 + depth pruning already, i.e. optimization work the authors already validated.

## D. HaMeR

```bash
conda create -n egohand-hamer python=3.10 -y && conda activate egohand-hamer
git clone --recursive https://github.com/geopavlakos/hamer.git
cd hamer && git rev-parse HEAD   # record this
python3.10 -m venv .hamer && source .hamer/bin/activate   # or reuse the conda env, repo supports either
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128  # match 01, not the repo's pinned cu117
pip install -e .[all]
pip install -v -e third-party/ViTPose
```
Note: the repo's README defaults to CUDA 11.7-era instructions — override with your `01` CUDA/
PyTorch combo, since Blackwell needs the newer stack regardless of what the repo assumes.

**MANO gate:** same registration as B/C.

In `models/hamer_wrapper.py`, implement `HamerHandModel(BaseHandModel)`:
- `load()`: use the repo's demo/checkpoint loading path, ViT-H backbone.
- `preprocess()`/`infer()`: mirror `demo.py`'s pipeline (detector → crop → HaMeR forward) closely.
- **Expect this one to be the most VRAM-hungry.** If it OOMs at batch size 1 on your 6GB effective
  budget, that is a valid, reportable result — try FP16 inference (`model.half()`) before giving
  up, note in `results/05_notes.md` whether FP16 alone was enough, and if not, benchmark it at
  reduced resolution or on a handful of images only, clearly labeled as "reduced protocol" in `06`.

---

## Per-model unit test (do this for all 4 before moving to `06`)

For each wrapper, run one forward pass on a single real FreiHAND image and assert:
- `preprocess()` doesn't throw.
- `infer()` returns a `list[HandPrediction]` with exactly one element for a single clear hand.
- `joints_3d.shape == (21, 3)`.
- For B/C/D: `mesh_verts.shape == (778, 3)` when present.

## Definition of Done

- [ ] All 4 wrappers exist, subclass `BaseHandModel` correctly, and pass their per-model unit test.
- [ ] `results/commit_hashes.json` has an entry for every cloned repo.
- [ ] MANO registration is confirmed complete (by the human user) before B/C/D unit tests are run.
- [ ] Any deviation from a repo's default instructions (e.g. HaMeR's CUDA version override) is
      written down in `results/05_notes.md` so it's not silently lost.
