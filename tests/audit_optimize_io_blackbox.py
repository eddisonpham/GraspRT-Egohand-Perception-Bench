"""Black-box audit of optimize/*.py CLI contracts and output JSON schemas.

No GPU needed: invokes --help (argparse contract) and validates the raw JSON
artifacts produced by prior real runs against their documented schema. Run:
    python tests/audit_optimize_io_blackbox.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    PASS += bool(cond)
    FAIL += not bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")


def run_help(script):
    r = subprocess.run([sys.executable, str(ROOT / script), "--help"],
                       capture_output=True, text=True, timeout=60)
    return r.returncode, r.stdout, r.stderr


print("=== BB-1: CLI --help exits 0 for all optimize scripts ===")
for script in ["optimize/export_onnx.py", "optimize/validate_onnx.py",
               "optimize/bench_onnx_latency.py", "optimize/build_trt_engine.py",
               "optimize/bench_trt.py", "optimize/build_trt_int8.py",
               "optimize/bench_trt_gate.py", "optimize/bench_breakdown.py"]:
    rc, out, err = run_help(script)
    check(f"{script} --help", rc == 0, f"rc={rc} err={err[:80]!r}")

print("=== BB-2: raw artifact schemas (from prior real runs) ===")
schemas = {
    "ort-latency.json": {"required": ["timestamp", "onnx", "variant", "results"],
                         "results_item_keys": {"mean_ms", "p95_ms", "fps", "n"}},
    "trt-latency.json": {"required": ["timestamp", "engine", "trt", "latency", "gate"],
                         "latency_keys": {"mean_ms", "p95_ms", "p99_ms", "fps"},
                         "gate_keys": {"joints_max_abs_diff", "pa_mpjpe_vs_pytorch_mm"}},
    "trt-accuracy-gate.json": {"required": ["timestamp", "per_engine", "gate"]},
    "latency-breakdown.json": {"required": ["timestamp", "model", "preprocess_plus_detector",
                                            "crop_collate", "model_forward", "postprocess",
                                            "sum_of_stages_ms"]},
}
for fname, spec in schemas.items():
    p = ROOT / "results" / "raw" / fname
    if not p.exists():
        check(f"{fname} schema", False, "missing")
        continue
    d = json.loads(p.read_text())
    ok = all(k in d for k in spec["required"])
    check(f"{fname} top-level keys", ok, f"missing={set(spec['required'])-set(d)}")
    if fname == "ort-latency.json" and d.get("results"):
        item = d["results"][0]
        check(f"{fname} results[0] has latency keys",
              spec["results_item_keys"].issubset(item), f"{set(item)}")
    if fname == "trt-latency.json":
        lat = d.get("latency", {})
        check(f"{fname} latency subkeys", spec["latency_keys"].issubset(lat), f"{set(lat)}")
        gate = d.get("gate", {})
        check(f"{fname} gate subkeys", spec["gate_keys"].issubset(gate), detail=f"{set(gate)}")

print("=== BB-3: validate_onnx exits cleanly without env (no silent crash) ===")
# Missing FREIHAND_ROOT must sys.exit with a message, not a traceback.
r = subprocess.run([sys.executable, str(ROOT / "optimize" / "validate_onnx.py")],
                   capture_output=True, text=True, timeout=60, env={**os.environ, "FREIHAND_ROOT": ""})
check("validate_onnx missing env -> nonzero exit", r.returncode != 0)
check("validate_onnx missing env -> message not traceback",
      "FREIHAND_ROOT" in (r.stdout + r.stderr) and "Traceback" not in (r.stdout + r.stderr))

print(f"\n=== SUMMARY: {PASS} passed, {FAIL} failed ===")
sys.exit(0 if FAIL == 0 else 1)
