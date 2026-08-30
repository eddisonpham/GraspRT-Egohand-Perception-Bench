# EgoHand-Bench: Benchmark & Optimize Hand Pose/Mesh Models for a 7GB-VRAM RTX 5060

This folder is a **linear runbook** for a coding agent (e.g. Claude Code) operating as a senior
ML research engineer. It implements, benchmarks, selects, and deploys the best egocentric-ready
hand pose/mesh estimation model under a hard **~7GB VRAM** budget on an **NVIDIA RTX 5060**
(Blackwell, `sm_120`), with a CPU fallback path available since the host has ample CPU cores.

This is the **perception module** of a larger "egocentric video → robot hand" pipeline. Scope is
deliberately bounded: pick the best hand pose/mesh backbone, prove it's fast and accurate under
your VRAM budget, and ship an optimized deployment artifact (ONNX + TensorRT engine). Retargeting
to a robot hand and Isaac Lab policy training are **out of scope** here — separate follow-on project.

## How the agent should use this folder

Process the numbered files **in strict order, 00 → 12**. Do not skip ahead. Each file ends with a
**Definition of Done** checklist — every item must be verified by actually running the specified
command and reading its output, not assumed from reading code. If a step fails, consult
`12_TROUBLESHOOTING.md` first; only improvise if the troubleshooting file doesn't cover it, and if
you do, append what you learned to `12_TROUBLESHOOTING.md` so the next run benefits.

Write all logs, raw benchmark JSON, plots, and intermediate artifacts into `results/` as you go —
`11_VALIDATION_SMOKE_TESTS.md` assembles the final report from those files, not from memory.

## File map

| # | File | What it produces |
|---|------|-------------------|
| 00 | `00_PROJECT_BRIEF.md` | Goals, constraints, definition of "done", inductive-bias reasoning |
| 01 | `01_ENVIRONMENT_SETUP.md` | Working CUDA/PyTorch/ONNX Runtime/TensorRT env on RTX 5060 |
| 02 | `02_LITERATURE_MODEL_SHORTLIST.md` | 4 candidate models, papers, repos, why each was picked |
| 03 | `03_REPO_SCAFFOLD.md` | Directory layout + shared `BaseHandModel` interface code |
| 04 | `04_DATASET_PREP.md` | FreiHAND quantitative eval set + your own egocentric qualitative clips |
| 05 | `05_MODEL_INTEGRATION.md` | Working wrapper for each of the 4 models |
| 06 | `06_BENCHMARK_HARNESS.md` | Latency / VRAM / accuracy numbers for all 4, raw JSON + plots |
| 07 | `07_DECISION_MATRIX.md` | Weighted scoring, chosen winner + justification |
| 08 | `08_OPTIMIZATION_ONNX_ORT.md` | ONNX export, ONNX Runtime CUDA/TensorRT EP benchmarks |
| 09 | `09_OPTIMIZATION_TENSORRT_QUANT.md` | Native TensorRT FP16 engine, optional INT8 PTQ w/ accuracy gate |
| 10 | `10_ADVANCED_CUDA_GRAPHS_TRITON.md` | *(optional/stretch)* CUDA Graphs capture, Triton serving |
| 11 | `11_VALIDATION_SMOKE_TESTS.md` | End-to-end smoke test + `results/FINAL_REPORT.md` |
| 12 | `12_TROUBLESHOOTING.md` | Known failure modes and fixes, especially `sm_120` issues |
| — | `REFERENCES.md` | Full bibliography — papers, repos, datasets |

## Non-negotiables

1. **Measure, don't assume.** Every claim in the final report must trace back to a number in
   `results/raw/*.json`.
2. **Stay under budget.** Target peak VRAM ≤ **6.0GB** during any single benchmark run (not 7GB) —
   leave headroom for OS/display compositor overhead, which is real on a laptop.
3. **One model loaded at a time** unless a step explicitly says otherwise — this is the single
   biggest lever for staying inside a 7GB card.
4. **Respect licenses.** FreiHAND is research-only/non-commercial. MANO requires a manual,
   click-through registration that cannot be automated — this is flagged where it matters.
