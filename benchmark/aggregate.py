"""Aggregate raw benchmark JSON into comparison / decision / Pareto artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"

COMPLEXITY = {"mediapipe": 1, "mobrecon": 3, "wilor": 3, "wilor-fast": 3, "hamer": 5}
WEIGHTS = {"latency": 0.35, "accuracy": 0.35, "vram": 0.20, "complexity": 0.10}
_WEIGHT_SUM = sum(WEIGHTS.values())


def load_rows() -> list[dict]:
    """Load benchmark rows, skipping malformed or non-benchmark JSON."""
    rows = []
    for p in sorted(RAW.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if "latency_ms" not in d or "accuracy" not in d:
            continue
        d["id"] = f"{d['model']}-{d.get('variant', 'default')}"
        rows.append(d)
    return rows


def normalize(vals, lower: bool = True) -> list[float]:
    """Min-max normalize to [0,1] (1.0 = best). None -> best (absent lower-is-better)."""
    valid = [v for v in vals if v is not None]
    if not valid:
        return [1.0] * len(vals)
    lo, hi = min(valid), max(valid)
    if hi == lo:
        return [1.0] * len(vals)
    out = []
    for v in vals:
        if v is None:
            out.append(1.0)
        else:
            n = (v - lo) / (hi - lo)
            out.append(1.0 - n if lower else n)
    return out


def _vram_norms(vram: list[float | None]) -> list[float]:
    """VRAM norm: absent (CPU) gets the median of present norms (neutral)."""
    present_n = normalize([v for v in vram if v is not None], lower=True)
    median_v = float(np.median(present_n)) if present_n else 0.5
    out = []
    for v in vram:
        if v is None:
            out.append(median_v)
            continue
        valid = [x for x in vram if x is not None]
        lo, hi = (min(valid), max(valid)) if valid else (0, 1)
        out.append(1.0 - (v - lo) / (hi - lo) if hi > lo else 1.0)
    return out


def _complexity_norm(key: str) -> tuple[int, float]:
    c = COMPLEXITY.get(key, 4)
    return c, 1.0 - (c - 1) / 4.0


def score_rows(rows: list[dict]) -> list[dict]:
    """Compute normalized metrics + weighted score for each row, in place."""
    lat = [r["latency_ms"]["mean"] for r in rows]
    acc = [r["accuracy"].get("pa_mpjpe_mm") for r in rows]
    vram = [r["vram_mb"].get("nvidia_smi_peak") for r in rows]

    lat_n = normalize(lat, lower=True)
    acc_n = [0.0 if r["accuracy"].get("pa_mpjpe_mm") is None else v
             for v, r in zip(normalize(acc, lower=True), rows)]
    vram_n = _vram_norms(vram)

    for i, r in enumerate(rows):
        key = r["model"] + ("-fast" if r.get("variant") == "fast" else "")
        c, c_n = _complexity_norm(key)
        score = (WEIGHTS["latency"] * (lat_n[i] or 0.0)
                 + WEIGHTS["accuracy"] * (acc_n[i] or 0.0)
                 + WEIGHTS["vram"] * (vram_n[i] or 0.0)
                 + WEIGHTS["complexity"] * c_n)
        r.update({
            "latency_norm": lat_n[i], "accuracy_norm": acc_n[i],
            "vram_norm": vram_n[i], "complexity": c, "complexity_norm": c_n,
            "score": score,
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


def _fmt_cell(v):
    return f"{v:.3f}" if v is not None else "—"


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
            f"| {r['fps']:.2f} | {_fmt_cell(a)} | {_fmt_cell(v)} "
            f"| {r.get('n_predictions', 0)}/{r.get('n_misses', 0)} "
            f"| {r['score']:.4f} |"
        )
    (ROOT / "results" / "comparison_table.md").write_text("\n".join(lines) + "\n")

    _write_pareto(rows)
    _write_decision(rows)
    print((ROOT / "results" / "comparison_table.md").read_text())
    print("wrote results/comparison_table.md, results/plots/pareto.png, results/decision.md")


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
        "# Decision Matrix\n",
        "## Weights",
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


if __name__ == "__main__":
    main()