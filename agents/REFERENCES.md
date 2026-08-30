# References

## Candidate models

- **MediaPipe Hands / Hand Landmarker** — Google.
  Repo: https://github.com/google-ai-edge/mediapipe
  Docs: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
  Model asset: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

- **MobRecon: Mobile-Friendly Hand Mesh Reconstruction from Monocular Image**
  Chen, X. et al. CVPR 2022.
  Paper: https://arxiv.org/abs/2112.02753
  Repo: https://github.com/SeanChenxy/HandMesh

- **WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild**
  Potamias, R.A., Zhang, J., Deng, J., Zafeiriou, S. CVPR 2025.
  Paper: https://arxiv.org/abs/2409.12259
  Repo: https://github.com/rolpotamias/WiLoR
  Project page: https://rolpotamias.github.io/WiLoR

- **Reconstructing Hands in 3D with Transformers (HaMeR)**
  Pavlakos, G., Shan, D., Radosavovic, I., Kanazawa, A., Fouhey, D., Malik, J. CVPR 2024.
  Paper: https://arxiv.org/abs/2312.05251
  Repo: https://github.com/geopavlakos/hamer

## Related work (context / forward-looking, not implemented)

- **Fast-HaMeR: Boosting Hand Mesh Reconstruction using Knowledge Distillation** (2026).
  Distills HaMeR's ViT-H backbone into MobileNet/MobileViT/ConvNeXt/ResNet students. Cited in
  `02` as context for why a distillation-based optimization path is plausible as future work.

- **MANO: Embodied Hands** — Romero, J., Tzionas, D., Black, M.J.
  Model registration (required by MobRecon, WiLoR, HaMeR): https://mano.is.tue.mpg.de/

## Datasets

- **FreiHAND: A Dataset for Markerless Capture of Hand Pose and Shape from Single RGB Images**
  Zimmermann, C., Ceylan, D., Yang, J., Russell, B., Argus, M., Brox, T. ICCV 2019.
  Project page: https://lmb.informatik.uni-freiburg.de/projects/freihand/
  Repo (loader/eval scripts): https://github.com/lmb-freiburg/freihand
  Download: https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip
  License: research/non-commercial use only.

## Optimization stack

- PyTorch CUDA Graphs: https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs
- ONNX Runtime execution providers: https://onnxruntime.ai/docs/execution-providers/
- TensorRT developer docs: https://docs.nvidia.com/deeplearning/tensorrt/
- Triton Inference Server: https://github.com/triton-inference-server/server

## Hardware/compatibility notes (RTX 5060 / Blackwell / `sm_120`)

- PyTorch tracking issue, official `sm_120` stable support:
  https://github.com/pytorch/pytorch/issues/159207 and
  https://github.com/pytorch/pytorch/issues/164342
- These were open/in-progress as of when this project plan was written — re-check current status
  before assuming which PyTorch channel (stable vs nightly) you need; this is exactly the kind of
  fast-moving compatibility detail that goes stale quickly.
