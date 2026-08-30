# Contributing to EgoHand-Bench

Thanks for helping improve the pipeline. The single governing rule of this repo:

> **Measure, don't assume.** No model is ranked, no optimization is claimed, and no
> result is reported without a real measured number in `results/raw/*.json`. If you
> can't measure it on your hardware, say so and open an issue instead of a PR with an
> estimate.

## Before you open a PR

1. **Run the full suite green:**
   ```bash
   python -m pytest tests/ -q
   python tests/audit_metrics_blackbox.py
   python tests/audit_aggregate_blackbox.py
   python tests/audit_optimize_io_blackbox.py
   # if FREIHAND_ROOT is set:
   python tests/audit_loader_blackbox.py
   ```
2. **Every new script must have an argparse CLI** so `--help` exits 0. The
   `audit_optimize_io_blackbox.py` executor enforces this — don't break it.
3. **New raw JSON artifacts must match their schema.** Add the schema to
   `tests/audit_optimize_io_blackbox.py` and verify.
4. **No assets are committed.** FreiHAND archives, WiLoR weights, MANO `.pkl`,
   ONNX/TRT engines, and external-data files are all in `.gitignore`. If you add a
   generated binary, extend `.gitignore`.

## Adding a model

1. Create `models/<name>_wrapper.py` implementing `common.BaseHandModel`
   (`load`, `preprocess`, `infer`, `warmup`, `device`).
2. Add it to `benchmark/run_benchmark.py`'s `--model` choices.
3. Add a `COMPLEXITY` entry in `benchmark/aggregate.py`.
4. Run `benchmark/run_benchmark.py --model <name> --iters 200 --subset dev` and
   commit the resulting `results/raw/<name>.json`.
5. If integration is blocked (missing deps, no wheel), add an explicit
   `hamer_wrapper.py`-style stub that raises with actionable diagnostics — do **not**
   fabricate a row.

## Code style

- `ruff` + `black`, line-length 100, target py310 (see `pyproject.toml`).
- Functions that return latency/accuracy numbers must document their units in the
  docstring (meters vs mm is a recurring footgun here — `pa_mpjpe` scales by 1000).

## Licensing

Your contributions are MIT-licensed (see `LICENSE`). You are responsible for **not**
committing restricted assets (see "Data / weights notice" in `README.md`).
