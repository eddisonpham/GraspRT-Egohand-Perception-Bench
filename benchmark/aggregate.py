"""Aggregate raw benchmark JSON into comparison / decision / Pareto artifacts.

The scoring core (`normalize`, `norms`, `score`) is kept free of I/O so the
eval suite can reuse the exact formula instead of re-implementing it.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"

COMPLEXITY = {"mediapipe": 1, "mobrecon": 3, "wilor": 3, "wilor-fast": 3, "hamer": 5}
WEIGHTS = {"latency": 0.35, "accuracy": 0.35, "vram": 0.20, "complexity": 0.10}


def normalize(vals: list, lower: bool = True) -> list[float]:
    """Min-max normalize to [0, 1] where 1.0 is best; absent (None) maps to 1.0."""
    valid = [v for v in vals if v is not None]
    if not valid:
        return [1.0] * len(vals)
    lo, hi = min(valid), max(valid)
    if hi == lo:
        return [1.0] * len(vals)
    return [1.0 if v is None else (1.0 - (v - lo) / (hi - lo) if lower
                                   else (v - lo) / (hi - lo)) for v in vals]


def complexity_norm(key: str) -> float:
    """Deployment-complexity sub-score from the COMPLEXITY map."""
    c = COMPLEXITY.get(key, 4)
    return 1.0 - (c - 1) / 4.0


def validate_norm(v: float | None) -> float:
    """Clamp a per-metric norm to [0, 1]; absent metrics score 0.0."""
    return 0.0 if v is None else max(0.0, min(1.0, v))


def score(lat_n: float, acc_n: float, vram_n: float, comp_n: float) -> float:
    """Weighted final score for one row."""
    return (WEIGHTS["latency"] * validate_norm(lat_n)
            + WEIGHTS["accuracy"] * validate_norm(acc_n)
            + WEIGHTS["vram"] * validate_norm(vram_n)
            + WEIGHTS["complexity"] * comp_n)


def vram_norms(vram: list[float | None]) -> tuple[list[float], float]:
    """VRAM norms + shared median for rows without a GPU measure (CPU models).

    A row without peak VRAM is neutral (gets the median of present norms).
    """
    present = normalize([v for v in vram if v is not None], lower=True)
    median_v = float(np.median(present)) if present else 0.5
    valid = [v for v in vram if v is not None]
    lo, hi = (min(valid), max(valid)) if valid else (0, 1)
    out = [median_v if v is None else (1.0 - (v - lo) / (hi - lo) if hi > lo else 1.0)
           for v in vram]
    return out, median_v


def metric_norms(rows: list[dict]) -> dict:
    """Per-row normalized latency/accuracy/VRAM/complexity plus final score."""
    lat = [r["latency_ms"]["mean"] for r in rows]
    acc = [r["accuracy"].get("pa_mpjpe_mm") for r in rows]
    vram = [r["vram_mb"].get("nvidia_smi_peak") for r in rows]
    lat_n = normalize(lat, lower=True)
    acc_n = normalize(acc, lower=True)
    vram_n, _ = vram_norms(vram)
    out = []
    for i, r in enumerate(rows):
        key = r["model"] + ("-fast" if r.get("variant") == "fast" else "")
        comp_n = complexity_norm(key)
        out.append({
            "latency_norm": lat_n[i], "accuracy_norm": acc_n[i],
            "vram_norm": vram_n[i], "complexity_norm": comp_n,
            "complexity": COMPLEXITY.get(key, 4),
            "score": score(lat_n[i], acc_n[i], vram_n[i], comp_n),
        })
    return {"metrics": out}


def load_rows() -> list[dict]:
    """Load benchmark rows from results/raw; skip malformed or non-benchmark JSON."""
    rows = []
    for p in sorted(RAW.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if "latency_ms" not in d or "accuracy" not in d:
            continue
        d["id"] = f"{d['model']}-{d.get('variant', 'default')}"
        rows.append(d)
    return rows


def score_rows(rows: list[dict]) -> list[dict]:
    """Attach normalized metrics + weighted score to each row, sorted best-first."""
    metrics = metric_norms(rows)["metrics"]
    for r, m in zip(rows, metrics):
        r.update(m)
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


def _fmt(v) -> str:
    return f"{v:.3f}" if v is not None else "—"


def _write_pareto(rows: list[dict]) -> None:
    plt.figure(figsize=(8, 5))
    for r in rows:
        x = r["accuracy"].get("pa_mpjpe_mm")
        y = r["latency_ms"].get("mean")
        if x is None or y is None:
            continue
        v = r["vram_mb"].get("nvidia_smi_peak") or 100
        plt.scatter(x, y, s=max(20, v / 5), alpha=0.8, label=r["id"])
    plt.xlabel("PA-MPJPE (mm, lower is better)")
    plt.ylabel("Mean latency (ms, lower is better)")
    plt.title("EgoHand-Bench Pareto Trade-off")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plot = ROOT / "results" / "plots" / "pareto.png"
    plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot, dpi=160)
    plt.close()


def _write_decision(rows: list[dict]) -> None:
    lines = [
        "# Decision Matrix\n", "## Weights",
        "- Latency mean: **0.35** (lower better)",
        "- PA-MPJPE: **0.35** (lower better)",
        "- Peak nvidia-smi VRAM: **0.20** (lower better)",
        "- Deployment complexity: **0.10** (lower better)\n",
        "## Ranked results\n",
        "| Rank | Candidate | Latency norm | Accuracy norm | VRAM norm | Complexity norm | Final score |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(f"| {i} | {r['id']} | {r['latency_norm']:.4f} "
                     f"| {r['accuracy_norm']:.4f} | {r['vram_norm']:.4f} "
                     f"| {r['complexity_norm']:.4f} | **{r['score']:.4f}** |")
    if rows:
        winner = rows[0]
        lines += ["", f"## Winner: **{winner['id']}**", "",
                  f"Selected by the stated weighted score ({winner['score']:.4f})."]
        if len(rows) > 1:
            runner = rows[1]
            lines += ["", f"## Runner-up: **{runner['id']}**", "",
                      "It wins only if its measured accuracy/latency/VRAM trade-off "
                      f"improves enough to exceed {winner['id']}'s score ({winner['score']:.4f})."]
    (ROOT / "results" / "decision.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    rows = load_rows()
    if not rows:
        print("No valid raw JSON files found")
        return
    rows = score_rows(rows)

    header = (
        "# Model Comparison\n\n"
        "Weights: latency 0.35, PA-MPJPE 0.35, nvidia-smi peak VRAM 0.20, "
        "deployment complexity 0.10. Missing metrics shown as —.\n\n"
        "| Model | Variant | Mean ms | p95 ms | FPS | PA-MPJPE mm | VRAM MB | "
        "Predictions/misses | Score |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = [header]
    for r in rows:
        a = r["accuracy"].get("pa_mpjpe_mm")
        v = r["vram_mb"].get("nvidia_smi_peak")
        lines.append(
            f"| {r['model']} | {r.get('variant', 'default')} "
            f"| {r['latency_ms']['mean']:.3f} | {r['latency_ms']['p95']:.3f} "
            f"| {r['fps']:.2f} | {_fmt(a)} | {_fmt(v)} "
            f"| {r.get('n_predictions', 0)}/{r.get('n_misses', 0)} "
            f"| {r['score']:.4f} |"
        )
    (ROOT / "results" / "comparison_table.md").write_text("\n".join(lines) + "\n")

    _write_pareto(rows)
    _write_decision(rows)
    print((ROOT / "results" / "comparison_table.md").read_text())
    print("wrote results/comparison_table.md, results/plots/pareto.png, results/decision.md")


if __name__ == "__main__":
    main()