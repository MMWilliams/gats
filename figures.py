#!/usr/bin/env python3
"""
Render publication-quality figures from the JSON output of ``reproduce.py``.

Produces PDF (for LaTeX) and PNG (for previewing) versions of:

  fig1_main_results.{pdf,png}     Success rate per method (Table 1)
  fig2_budget_ablation.{pdf,png}  GATS performance vs search budget (Table 2)
  fig3_llm_calls.{pdf,png}        LLM calls / task — cost comparison
  fig4_stress_test.{pdf,png}      Per-category stress-test heatmap (Table 6)
  fig5_world_model.{pdf,png}      World-model layer ablation (Table 3)

Also writes:
  results/summary.csv             tidy-format table for downstream tooling
  results/tables.tex              LaTeX booktabs versions of the key tables

Run standalone:
    python figures.py --results-dir results
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Publication style: serif fonts, vector output, tight margins.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Stable colour palette (colour-blind safe, matches paper convention)
COLORS = {
    "greedy":   "#4D4D4D",
    "react":    "#E69F00",
    "lats":     "#56B4E9",
    "lats_b5":  "#A6CEE3",
    "lats_b10": "#56B4E9",
    "lats_b20": "#1F78B4",
    "gats":     "#009E73",
    "gats_b1":  "#B2DF8A",
    "gats_b5":  "#66C2A5",
    "gats_b10": "#009E73",
    "gats_b20": "#006D5B",
    "gats_b50": "#003D33",
}

LLM_CALLS_PER_TASK = {
    # Paper Table 1: number of LLM inference calls per task during planning.
    "greedy":   0,
    "react":    13,
    "lats_b5":  17,
    "lats_b10": 37,
    "gats_b5":  0,
    "gats_b10": 0,
    "gats_b20": 0,
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    plt.close(fig)
    print(f"  wrote {out.name}/{name}.{{pdf,png}}")


def _mean_metric(runs: list[dict], key: str) -> float:
    if not runs:
        return 0.0
    return float(np.mean([r[key] for r in runs]))


def _color_for(method: str) -> str:
    if method in COLORS:
        return COLORS[method]
    for prefix in ("gats", "lats"):
        if method.startswith(prefix):
            return COLORS[prefix]
    return "#888888"


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_main_results(main_data: dict, out: Path):
    """Bar chart: success rate per method (paper Table 1)."""
    order = ["react", "lats_b5", "lats_b10", "gats_b5", "gats_b10", "gats_b20"]
    labels = ["ReAct", "LATS (b=5)", "LATS (b=10)",
              "GATS (b=5)", "GATS (b=10)", "GATS (b=20)"]
    present = [(m, l) for m, l in zip(order, labels) if m in main_data]
    if not present:
        return

    means = [_mean_metric(main_data[m], "success_rate") * 100 for m, _ in present]
    stds = [float(np.std([r["success_rate"] for r in main_data[m]])) * 100
            for m, _ in present]
    colors = [_color_for(m) for m, _ in present]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = np.arange(len(present))
    bars = ax.bar(x, means, yerr=stds, color=colors,
                  capsize=4, edgecolor="black", linewidth=0.6)

    ax.axhline(100, color="#888", linestyle="--", linewidth=0.8, zorder=0)
    ax.text(-0.45, 100, "oracle 100%", color="#666", fontsize=9,
            ha="left", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in present], rotation=15, ha="right")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Main evaluation — 100 synthetic planning tasks")
    ax.set_ylim(0, 110)

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                f"{val:.1f}", ha="center", fontsize=9)

    _save(fig, out, "fig1_main_results")


def fig_budget_ablation(main_data: dict, out: Path):
    """Line plot: GATS success rate vs search budget (Table 2)."""
    budgets, srs, opts = [], [], []
    for key in sorted(main_data.keys(), key=lambda k: int(k.split("_b")[-1])
                      if k.startswith("gats_b") else -1):
        if not key.startswith("gats_b"):
            continue
        try:
            b = int(key.split("_b")[-1])
        except ValueError:
            continue
        budgets.append(b)
        srs.append(_mean_metric(main_data[key], "success_rate") * 100)
        opts.append(_mean_metric(main_data[key], "optimality"))
    if not budgets:
        return

    fig, ax1 = plt.subplots(figsize=(6.0, 3.6))
    ax1.plot(budgets, srs, "o-", color=COLORS["gats"], linewidth=2.0,
             markersize=7, label="Success rate")
    ax1.set_xlabel("Search budget $b$")
    ax1.set_ylabel("Success rate (%)", color=COLORS["gats"])
    ax1.tick_params(axis="y", labelcolor=COLORS["gats"])
    ax1.set_ylim(0, 110)
    ax1.set_xscale("log")
    ax1.set_xticks(budgets)
    ax1.set_xticks([], minor=True)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax2 = ax1.twinx()
    ax2.plot(budgets, opts, "s--", color="#D55E00", linewidth=1.5,
             markersize=6, label="Optimality")
    ax2.set_ylabel("Optimality", color="#D55E00")
    ax2.tick_params(axis="y", labelcolor="#D55E00")
    ax2.set_ylim(0, 1.1)
    ax2.spines["top"].set_visible(False)

    for b, sr in zip(budgets, srs):
        ax1.annotate(f"{sr:.0f}%", (b, sr), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=9,
                     color=COLORS["gats"])

    ax1.set_title("GATS: budget scaling")
    fig.tight_layout()
    _save(fig, out, "fig2_budget_ablation")


def fig_llm_calls(main_data: dict, out: Path):
    """Cost comparison: LLM calls per task across methods."""
    methods, calls, srs, colors = [], [], [], []
    for m in ["react", "lats_b5", "lats_b10", "gats_b10", "gats_b20"]:
        if m not in main_data:
            continue
        methods.append(m)
        calls.append(LLM_CALLS_PER_TASK.get(m, 0))
        srs.append(_mean_metric(main_data[m], "success_rate") * 100)
        colors.append(_color_for(m))
    if not methods:
        return

    label_map = {"react": "ReAct", "lats_b5": "LATS (b=5)",
                 "lats_b10": "LATS (b=10)", "gats_b10": "GATS (b=10)",
                 "gats_b20": "GATS (b=20)"}

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    sizes = [80 + 6 * s for s in srs]
    ax.scatter(calls, srs, s=sizes, c=colors, alpha=0.85,
               edgecolors="black", linewidths=0.7, zorder=3)
    # When two markers share an x, stack their labels vertically.
    label_offset = {"gats_b20": (12, 10), "gats_b10": (12, -16)}
    for m, c, s in zip(methods, calls, srs):
        dx, dy = label_offset.get(m, (10, 4))
        ax.annotate(label_map[m], (c, s), xytext=(dx, dy),
                    textcoords="offset points", fontsize=10)
    ax.set_xlabel("LLM inference calls per task (planning)")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Cost vs. performance — GATS dominates the Pareto front")
    ax.set_xlim(-3, max(calls) * 1.25 + 1 if calls else 1)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.25, zorder=0)
    _save(fig, out, "fig3_llm_calls")


def fig_stress_test(stress_data: dict, out: Path):
    """Heatmap: success rate by category x method (Table 6)."""
    methods = [m for m in ["react", "lats_b10", "lats_b20",
                           "gats_b10", "gats_b20"] if m in stress_data]
    if not methods:
        return
    cats = sorted({c for m in methods for c in stress_data[m]})
    if not cats:
        return

    matrix = np.zeros((len(methods), len(cats)))
    for i, m in enumerate(methods):
        for j, c in enumerate(cats):
            matrix[i, j] = stress_data[m].get(c, 0.0) * 100

    fig, ax = plt.subplots(figsize=(10.5, 3.5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(cats)))
    ax.set_xticklabels(cats, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([m.replace("_", " ") for m in methods])
    for i in range(len(methods)):
        for j in range(len(cats)):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    color="black" if 25 <= val <= 75 else "white",
                    fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Success rate (%)")
    ax.set_title("Stress test — 12 challenging planning categories")
    _save(fig, out, "fig4_stress_test")


def fig_world_model_ablation(main_data: dict, out: Path):
    """Bar chart: layered world-model ablation (Table 3)."""
    keys = [("gats_b10", "GATS (full)"),
            ("gats_no_l1", "no L1 (symbolic)"),
            ("gats_no_l3", "no L3 (LLM)")]
    present = [(k, l) for k, l in keys if k in main_data]
    if len(present) < 2:
        return

    means = [_mean_metric(main_data[k], "success_rate") * 100
             for k, _ in present]
    opts = [_mean_metric(main_data[k], "optimality") for k, _ in present]
    x = np.arange(len(present))
    width = 0.36

    fig, ax1 = plt.subplots(figsize=(5.6, 3.6))
    b1 = ax1.bar(x - width / 2, means, width, color=COLORS["gats"],
                 label="Success rate", edgecolor="black", linewidth=0.6)
    ax2 = ax1.twinx()
    b2 = ax2.bar(x + width / 2, opts, width, color="#D55E00",
                 label="Optimality", edgecolor="black", linewidth=0.6)
    ax2.spines["top"].set_visible(False)

    for rect, v in zip(b1, means):
        ax1.text(rect.get_x() + rect.get_width() / 2, v + 1.5,
                 f"{v:.0f}%", ha="center", fontsize=9, color=COLORS["gats"])
    for rect, v in zip(b2, opts):
        ax2.text(rect.get_x() + rect.get_width() / 2, v + 0.02,
                 f"{v:.2f}", ha="center", fontsize=9, color="#D55E00")

    ax1.set_xticks(x)
    ax1.set_xticklabels([l for _, l in present])
    ax1.set_ylabel("Success rate (%)", color=COLORS["gats"])
    ax2.set_ylabel("Optimality", color="#D55E00")
    ax1.set_ylim(0, 115)
    ax2.set_ylim(0, 1.15)
    ax1.set_title("World-model layer ablation")
    fig.tight_layout()
    _save(fig, out, "fig5_world_model")


# ---------------------------------------------------------------------------
# CSV + LaTeX summaries
# ---------------------------------------------------------------------------

def write_summary_csv(main_data: dict, stress_data: dict, out: Path):
    rows = [["source", "method", "metric", "value"]]
    for m, runs in (main_data or {}).items():
        for metric in ("success_rate", "optimality", "avg_cost",
                       "avg_plan_length", "avg_nodes_expanded"):
            rows.append(["main_eval", m, metric, f"{_mean_metric(runs, metric):.6f}"])
    for m, cats in (stress_data or {}).items():
        for cat, sr in cats.items():
            rows.append(["stress_test", m, f"sr/{cat}", f"{sr:.6f}"])
        if cats:
            rows.append(["stress_test", m, "sr/overall",
                         f"{sum(cats.values()) / len(cats):.6f}"])

    out.write_text("\n".join(",".join(r) for r in rows) + "\n")
    print(f"  wrote {out.name}")


def write_latex_tables(main_data: dict, stress_data: dict, out: Path):
    lines = [r"% Auto-generated — GATS paper tables.", ""]

    # Table 1: main results
    if main_data:
        lines += [
            r"\begin{table}[t]\centering",
            r"\caption{Main evaluation on synthetic multi-step planning tasks.}",
            r"\label{tab:main}",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Method & Success Rate & Optimality & Avg Cost & LLM Calls \\",
            r"\midrule",
        ]
        for m in ["greedy", "react", "lats_b5", "lats_b10",
                  "gats_b5", "gats_b10", "gats_b20"]:
            if m not in main_data:
                continue
            sr = _mean_metric(main_data[m], "success_rate") * 100
            opt = _mean_metric(main_data[m], "optimality")
            cost = _mean_metric(main_data[m], "avg_cost")
            calls = LLM_CALLS_PER_TASK.get(m, "--")
            label = m.replace("_", "\\_")
            lines.append(f"  {label} & {sr:.1f}\\% & {opt:.2f} & {cost:.2f} & {calls} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # Table 6: stress test overall
    if stress_data:
        lines += [
            r"\begin{table}[t]\centering",
            r"\caption{Stress test — overall success rate over 12 categories.}",
            r"\label{tab:stress}",
            r"\begin{tabular}{lr}",
            r"\toprule",
            r"Method & Overall Success Rate \\",
            r"\midrule",
        ]
        for m in ["react", "lats_b10", "lats_b20", "gats_b10", "gats_b20"]:
            if m not in stress_data:
                continue
            cats = stress_data[m]
            if not cats:
                continue
            overall = sum(cats.values()) / len(cats) * 100
            label = m.replace("_", "\\_")
            lines.append(f"  {label} & {overall:.1f}\\% \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    out.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out.name}")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="results",
                   help="Directory containing main_eval.json + stress_test.json")
    args = p.parse_args()

    results = Path(args.results_dir).resolve()
    figs = results / "figures"

    main_path = results / "main_eval.json"
    stress_path = results / "stress_test.json"

    main_data = json.loads(main_path.read_text()) if main_path.exists() else {}
    stress_data = json.loads(stress_path.read_text()) if stress_path.exists() else {}

    if not main_data and not stress_data:
        print(f"No results found in {results}. Run `python reproduce.py` first.")
        return 1

    if main_data:
        fig_main_results(main_data, figs)
        fig_budget_ablation(main_data, figs)
        fig_llm_calls(main_data, figs)
        fig_world_model_ablation(main_data, figs)
    if stress_data:
        fig_stress_test(stress_data, figs)

    write_summary_csv(main_data, stress_data, results / "summary.csv")
    write_latex_tables(main_data, stress_data, results / "tables.tex")

    print(f"\n  All figures in: {figs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
