# Model Comparison

Weights: latency 0.35, PA-MPJPE 0.35, nvidia-smi peak VRAM 0.20, deployment complexity 0.10.
Missing metrics are shown as `—`; only actual benchmark JSON values are included.

| Model | Variant | Mean ms | p95 ms | FPS | PA-MPJPE mm | VRAM MB | Predictions/misses | Score |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| wilor | fast | 35.622 | 41.838 | 28.07 | 5.903 | 3363.0 | 200/0 | 0.6942 |
| mediapipe | default | 34.351 | 46.503 | 29.11 | 15.883 | — | 180/20 | 0.6364 |
| mobrecon | default | 19.760 | 34.457 | 50.61 | 35.665 | 1210.0 | 200/0 | 0.6000 |
| wilor | default | 69.531 | 77.272 | 14.38 | 5.742 | 4235.0 | 200/0 | 0.4000 |
