# Decision Matrix

## Weights

- Latency mean: **0.35** (lower better)
- PA-MPJPE: **0.35** (lower better)
- Peak nvidia-smi VRAM: **0.20** (lower better)
- Deployment complexity: **0.10** (lower better)

## Ranked results

| Rank | Candidate | Latency norm | Accuracy norm | VRAM norm | Complexity norm | Final score |
|---:|---|---:|---:|---:|---:|---:|
| 1 | wilor-fast | 0.6813 | 0.9946 | 0.2883 | 0.5000 | **0.6942** |
| 2 | mediapipe-default | 0.7068 | 0.6611 | 0.2883 | 1.0000 | **0.6364** |
| 3 | mobrecon-default | 1.0000 | 0.0000 | 1.0000 | 0.5000 | **0.6000** |
| 4 | wilor-default | 0.0000 | 1.0000 | 0.0000 | 0.5000 | **0.4000** |

## Winner: **wilor-fast**

Selected by the stated weighted score (0.6942), not by an after-the-fact preference.

## Runner-up: **mediapipe-default**

It would win if its measured accuracy/latency/VRAM trade-off improved enough to exceed the winner's weighted score.
