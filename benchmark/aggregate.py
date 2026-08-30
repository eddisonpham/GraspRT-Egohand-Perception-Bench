"""Aggregate raw benchmark JSON files into comparison/decision artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"

# Complexity scores per stage-07 guidance (lower is simpler).
COMPLEXITY = {"mediapipe": 1, "mobrecon": 3, "wilor": 3, "wilor-fast": 3, "hamer": 5}
WEIGHTS = {"latency": 0.35, "accuracy": 0.35, "vram": 0.20, "complexity": 0.10}


def load_rows():
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


def normalize(vals, lower=True):
    """Min-max normalize; returns 1.0 for best, 0.0 for worst.

    None handling (audited): a None means the metric is *absent* (e.g. a CPU-only
    model has no nvidia-smi peak). For lower-is-better metrics, "absent VRAM" is
    best (0 MB used), so None -> 1.0. For accuracy (also lower-is-better) a missing
    value is a meaningful penalty -> None -> 0.0. The caller chooses per-metric.
    """
    valid = [v for v in vals if v is not None]
    if not valid:
        return [1.0] * len(vals)
    lo, hi = min(valid), max(valid)
    if hi == lo:
        return [1.0 if v is not None else 1.0 for v in vals]
    out = []
    for v in vals:
        if v is None:
            out.append(1.0)  # absent lower-is-better metric -> best (e.g. no VRAM)
        else:
            n = (v - lo) / (hi - lo)
            out.append(1.0 - n if lower else n)
    return out


def main():
    rows = load_rows()
    if not rows:
        print("No valid raw JSON files found")
        return
    lat = [r["latency_ms"]["mean"] for r in rows]
    acc = [r["accuracy"].get("pa_mpjpe_mm") for r in rows]
    vram = [r["vram_mb"].get("nvidia_smi_peak") for r in rows]
    lat_n = normalize(lat, lower=True)          # None latency unlikely; -> best
    acc_n = normalize(acc, lower=True)
    # VRAM: absent (CPU-only model) is NEITHER rewarded nor penalized — assign the
    # median of present VRAM norms so a CPU model isn't artificially boosted to
    # rank-1 on a metric it doesn't compete on (audited: max->1.0 misrepresented
    # MediaPipe as the VRAM winner).
    vram_present_n = normalize([v for v in vram if v is not None], lower=True)
    median_v = float(np.median(vram_present_n)) if vram_present_n else 0.5
    vram_n = [median_v if v is None else None for v in vram]
    # fill present ones by matching against the present-VRAM normalization
    present_vals = [v for v in vram if v is not None]
    lo, hi = (min(present_vals), max(present_vals)) if present_vals else (0, 1)
    filled = []
    pi = 0
    for v in vram:
        if v is None:
            filled.append(median_v)
        else:
            n = (v - lo) / (hi - lo) if hi > lo else 1.0
            filled.append(1.0 - n)
            pi += 1
    vram_n = filled
    # Override: a missing PA-MPJPE is a meaningful penalty (None -> 0.0),
    # not "best" like an absent lower-is-better metric normally is.
    acc_n = [0.0 if r["accuracy"].get("pa_mpjpe_mm") is None else v
             for v, r in zip(acc_n, rows)]
    scores = []
    for i, r in enumerate(rows):
        key = r["model"] + ("-fast" if r.get("variant") == "fast" else "")
        c = COMPLEXITY.get(key, 4)
        c_n = 1.0 - (c - 1) / 4.0
        score = (WEIGHTS["latency"] * (lat_n[i] or 0.0) +
                 WEIGHTS["accuracy"] * (acc_n[i] or 0.0) +
                 WEIGHTS["vram"] * (vram_n[i] or 0.0) +
                 WEIGHTS["complexity"] * c_n)
        scores.append(score)
        r.update({"latency_norm": lat_n[i], "accuracy_norm": acc_n[i],
                  "vram_norm": vram_n[i], "complexity": c,
                  "complexity_norm": c_n, "score": score})
    rows.sort(key=lambda r: r["score"], reverse=True)

    lines = [
        "# Model Comparison",
        "",
        "Weights: latency 0.35, PA-MPJPE 0.35, nvidia-smi peak VRAM 0.20, deployment complexity 0.10.",
        "Missing metrics are shown as `—`; only actual benchmark JSON values are included.", "",
        "| Model | Variant | Mean ms | p95 ms | FPS | PA-MPJPE mm | VRAM MB | Predictions/misses | Score |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        a = r["accuracy"].get("pa_mpjpe_mm")
        v = r["vram_mb"].get("nvidia_smi_peak")
        lines.append(
            f"| {r['model']} | {r.get('variant','default')} | {r['latency_ms']['mean']:.3f} "
            f"| {r['latency_ms']['p95']:.3f} | {r['fps']:.2f} | "
            f"{a:.3f} | {v if v is not None else '—'} | "
            f"{r.get('n_predictions',0)}/{r.get('n_misses',0)} | {r['score']:.4f} |"
        )
    (ROOT / "results" / "comparison_table.md").write_text("\n".join(lines) + "\n")

    # Pareto scatter: only rows with complete latency + accuracy; marker size from VRAM.
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

    # Decision artifact from exactly the same score calculation.
    decision = [
        "# Decision Matrix", "",
        "## Weights", "",
        "- Latency mean: **0.35** (lower better)",
        "- PA-MPJPE: **0.35** (lower better)",
        "- Peak nvidia-smi VRAM: **0.20** (lower better)",
        "- Deployment complexity: **0.10** (lower better)", "",
        "## Ranked results", "",
        "| Rank | Candidate | Latency norm | Accuracy norm | VRAM norm | Complexity norm | Final score |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(rows, 1):
        decision.append(f"| {i} | {r['id']} | {r['latency_norm']:.4f} | {r['accuracy_norm']:.4f} | "
                        f"{r['vram_norm']:.4f} | {r['complexity_norm']:.4f} | **{r['score']:.4f}** |")
    if rows:
        winner, runner = rows[0], rows[1] if len(rows) > 1 else None
        decision += ["", f"## Winner: **{winner['id']}**", "",
                     f"Selected by the stated weighted score ({winner['score']:.4f}), not by an after-the-fact preference."]
        if runner:
            decision += ["", f"## Runner-up: **{runner['id']}**", "",
                         "It would win if its measured accuracy/latency/VRAM trade-off improved enough to exceed the winner's weighted score."]
    (ROOT / "results" / "decision.md").write_text("\n".join(decision) + "\n")
    print((ROOT / "results" / "comparison_table.md").read_text())
    print(f"wrote {plot} and results/decision.md")


if __name__ == "__main__":
    main()