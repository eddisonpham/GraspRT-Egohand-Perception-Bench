## Summary
<!-- What this PR changes, in one or two sentences. -->

## Why
<!-- The motivation. Link any issue: "Closes #12". -->

## Measured impact
<!-- For any change to benchmark/metrics/optimization: paste the BEFORE and AFTER
numbers from results/raw/*.json. If this PR does not affect measured outputs,
say "no measured change". No claim is accepted without numbers or a code-only justification. -->

| Metric | Before | After | Delta |
|---|---|---|---|
| | | | |

## Tests
- [ ] `python -m pytest tests/ -q` passes
- [ ] `python tests/audit_metrics_blackbox.py` passes
- [ ] `python tests/audit_aggregate_blackbox.py` passes
- [ ] `python tests/audit_optimize_io_blackbox.py` passes
- [ ] (If data present) `python tests/audit_loader_blackbox.py` passes

## Reproducibility checklist
- [ ] No fake/estimated numbers introduced
- [ ] New raw JSON artifacts follow the documented schema (run `audit_optimize_io_blackbox.py`)
- [ ] New scripts have an argparse CLI and `--help` exits 0
- [ ] `.gitignore` updated if new large/generated files are produced
