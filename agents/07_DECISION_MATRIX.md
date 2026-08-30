# 07 — Decision Matrix

Turn the `results/raw/*.json` numbers from `06` into one defensible winner, with the reasoning
written down — not just a gut call on the Pareto plot.

## Default weights (adjust if your priorities differ, but state that you did)

| Criterion | Weight | Direction |
|---|---|---|
| Latency (mean, ms) | 0.35 | lower is better |
| Accuracy (PA-MPJPE, mm) | 0.35 | lower is better |
| Peak VRAM (nvidia-smi, MB) | 0.20 | lower is better |
| Deployment complexity (1–5, subjective) | 0.10 | lower is better |

These weights reflect the project's actual constraint: this is an edge-deployment project first,
a maximum-accuracy project second. If your priorities differ (e.g. you'd trade more latency for
meaningfully better accuracy), change the weights here and say so explicitly — don't silently
re-weight after seeing which model "should" win.

**Deployment complexity** is the one subjective input — score it honestly per model, e.g.: does it
need a separate detector stage, custom preprocessing, unusual dependencies (torch-geometric,
detectron2-style detectors), or license friction. MediaPipe = 1 (pip install, done). HaMeR is
probably a 4–5 (ViT-H, heavier dependency chain, MANO gate). Write one sentence justifying each
score in `results/07_notes.md`.

## Scoring procedure

1. For each criterion, min-max normalize across all candidates to `[0, 1]`, where 1 = best:
   ```python
   def normalize(values, lower_is_better=True):
       lo, hi = min(values), max(values)
       if hi == lo:
           return [1.0 for _ in values]
       normed = [(v - lo) / (hi - lo) for v in values]
       return [1 - n for n in normed] if lower_is_better else normed
   ```
2. Weighted sum per candidate: `score = 0.35*latency_norm + 0.35*accuracy_norm + 0.20*vram_norm + 0.10*(1 - (complexity-1)/4)`.
3. Rank candidates by `score`, descending.

## Tie-break rules (apply in this order if scores are within 0.02 of each other)

1. Prefer the model that also produces a mesh (MANO output) over one that only gives keypoints —
   this pipeline's downstream use (retargeting) needs it, per inductive bias #3 in `00`.
2. Prefer the model with lower deployment complexity.
3. Prefer the model with better p95 (tail) latency, not just mean — matters more for a real-time
   control-loop-adjacent use case than a marginal mean-latency difference.

## Output: `results/decision.md`

Must contain:
1. The full scored table (one row per model/variant, all 4 normalized sub-scores + final weighted
   score, sorted).
2. **The chosen winner**, stated explicitly.
3. **The runner-up**, and one sentence on what would need to change for it to win instead (this
   proves you understand the trade-off rather than just reading off a formula).
4. One paragraph connecting the result back to the inductive biases stated in `00` — did reality
   agree with the priors, or not? A prior turning out wrong is a fine, reportable outcome as long
   as you say so plainly.

The winner (and, optionally, the runner-up for comparison) is what proceeds to `08`/`09`.

## Definition of Done

- [ ] `results/decision.md` exists with the full scored table and an explicit winner + runner-up.
- [ ] Weights used are stated, and if changed from the defaults, the reason is stated too.
- [ ] The paragraph connecting results back to `00`'s inductive biases is written and specific
      (references actual numbers), not generic.
