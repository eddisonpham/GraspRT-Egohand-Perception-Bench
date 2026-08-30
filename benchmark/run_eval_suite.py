"""Fixed evaluation suite: one command to validate the full pipeline.

Runs in three passes (all no-GPU):
  1. pytest unit + edge-case + profiling tests
  2. Raw JSON schema + aggregate scoring validation
  3. Writes results/eval-report.md with pass/fail summary

Usage:
  python benchmark/run_eval_suite.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results"

RESULTS_SCHEMA_REQUIRED = {"latency_ms", "accuracy", "vram_mb"}


def run_tests() -> dict:
    """Run pytest on the no-GPU suite; return structured result."""
    import subprocess
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q",
         "--tb=short", "-x"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    wall_ms = (time.perf_counter() - t0) * 1000
    timed = [l for l in proc.stdout.strip().splitlines() if "passed" in l]
    summary = timed[-1] if timed else proc.stdout.strip().splitlines()[-1]
    return {
        "exit_code": proc.returncode,
        "wall_ms": round(wall_ms, 1),
        "summary": summary.strip(),
        "passed": proc.returncode == 0,
    }


def validate_raw_jsons() -> dict:
    """Check every results/raw/*.json parses and has the benchmark schema when applicable."""
    raw_dir = OUT / "raw"
    results = []
    if not raw_dir.exists():
        return {"n_files": 0, "n_valid": 0, "n_schema_ok": 0, "errors": [], "passed": True}
    errors = []
    n_valid = n_schema_ok = 0
    for p in sorted(raw_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text())
            n_valid += 1
        except Exception as e:
            errors.append(f"{p.name}: parse error: {e}")
            continue
        is_benchmark = "latency_ms" in d and "accuracy" in d
        if is_benchmark:
            missing = RESULTS_SCHEMA_REQUIRED - set(d.keys())
            if missing:
                errors.append(f"{p.name}: missing benchmark keys: {missing}")
            else:
                n_schema_ok += 1
                if "mean" not in d["latency_ms"]:
                    errors.append(f"{p.name}: latency_ms.mean missing")
                if "pa_mpjpe_mm" not in d["accuracy"]:
                    errors.append(f"{p.name}: accuracy.pa_mpjpe_mm missing")
        else:
            n_schema_ok += 1
        results.append(p.name)
    return {
        "n_files": len(results),
        "n_valid": n_valid,
        "n_schema_ok": n_schema_ok,
        "errors": errors,
        "passed": len(errors) == 0,
    }


def validate_aggregate() -> dict:
    """Recompute scores from the shared aggregate core and check consistency."""
    from benchmark.aggregate import load_rows, metric_norms, score
    rows = load_rows()
    if not rows:
        return {"n_rows": 0, "errors": [], "passed": True}
    metrics = metric_norms(rows)["metrics"]
    errors = []
    for r, m in zip(rows, metrics):
        fresh = score(m["latency_norm"], m["accuracy_norm"], m["vram_norm"],
                      m["complexity_norm"])
        if r.get("score") is not None and abs(r["score"] - fresh) > 0.01:
            errors.append(f"{r['id']}: stored {r['score']:.4f} vs recomputed {fresh:.4f}")
    return {"n_rows": len(rows), "errors": errors, "passed": len(errors) == 0}


def validate_artifacts() -> dict:
    """Check that key result artifacts exist and are non-empty."""
    checks = [
        ("results/comparison_table.md", "comparison table"),
        ("results/decision.md", "decision matrix"),
    ]
    results = []
    errors = []
    for rel, label in checks:
        p = ROOT / rel
        exists = p.exists() and p.stat().st_size > 100
        results.append({"path": rel, "exists": exists})
        if not exists:
            errors.append(f"{label} missing or empty ({rel})")
    return {"checks": results, "errors": errors, "passed": len(errors) == 0}


def main() -> None:
    t0 = time.perf_counter()
    sections = {}

    print("=" * 60)
    print("PASS 1: Running pytest suite...")
    print("=" * 60)
    sections["tests"] = run_tests()
    status = "PASS" if sections["tests"]["passed"] else "FAIL"
    print(f"  {status}: {sections['tests']['summary']}")

    print("\n" + "=" * 60)
    print("PASS 2: Validating raw JSONs and aggregate...")
    print("=" * 60)
    sections["raw_jsons"] = validate_raw_jsons()
    rj = sections["raw_jsons"]
    print(f"  {rj['n_files']} files, {rj['n_valid']} valid, {rj['n_schema_ok']} schema-ok")
    for e in rj["errors"]:
        print(f"  ERROR: {e}")

    sections["aggregate"] = validate_aggregate()
    agg = sections["aggregate"]
    print(f"  Aggregate: {agg['n_rows']} rows")
    for e in agg["errors"]:
        print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print("PASS 3: Checking artifacts...")
    print("=" * 60)
    sections["artifacts"] = validate_artifacts()
    for e in sections["artifacts"]["errors"]:
        print(f"  ERROR: {e}")

    wall_ms = (time.perf_counter() - t0) * 1000
    all_pass = all(
        s.get("passed", False) if isinstance(s, dict) else True
        for s in sections.values()
    )
    overall = "PASS" if all_pass else "FAIL"
    print(f"\n{'=' * 60}")
    print(f"OVERALL: {overall} ({wall_ms:.0f} ms wall time)")
    print(f"{'=' * 60}")

    OUT.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Eval Report", "",
        f"**Status**: {overall}  ",
        f"**Wall time**: {wall_ms:.0f} ms  ",
        f"**Platform**: {os.uname().nodename if hasattr(os, 'uname') else os.name}", "",
        "## Test suite", "",
        f"- {sections['tests']['summary']}",
        f"- Exit code: {sections['tests']['exit_code']}", "",
        "## Raw JSON validation", "",
        f"- {rj['n_files']} files, {rj['n_valid']} parseable, {rj['n_schema_ok']} schema-correct",
    ]
    if rj["errors"]:
        lines.append("")
        for e in rj["errors"]:
            lines.append(f"- ERROR: {e}")
    lines += [
        "",
        "## Aggregate scoring", "",
        f"- {agg['n_rows']} benchmark rows scored",
    ]
    if agg["errors"]:
        lines.append("")
        for e in agg["errors"]:
            lines.append(f"- ERROR: {e}")
    lines += [
        "",
        "## Artifact checks", "",
    ]
    for c in sections["artifacts"]["checks"]:
        icon = "[OK]" if c["exists"] else "[MISSING]"
        lines.append(f"- {icon} {c['path']}")
    if sections["artifacts"]["errors"]:
        lines.append("")
        for e in sections["artifacts"]["errors"]:
            lines.append(f"- ERROR: {e}")
    report = OUT / "eval-report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written to {report}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
