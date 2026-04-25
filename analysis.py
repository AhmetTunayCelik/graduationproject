"""
analysis.py
============

Academic-grade analysis and visualisation for MAX-APC experiments.

Loads all heuristic results and Gurobi optimal solutions from the results/
directory, computes performance metrics, and produces:

  Excel workbook  (analysis_output/analysis_tables.xlsx)
  ├── Raw_Data           — every individual run (full replication data)
  ├── Summary_by_Algo    — mean gap, runtime, feasibility% per algorithm
  ├── Gap_Table          — pivot: (n, β) rows × algorithm columns
  ├── Runtime_Table      — mean runtime per n for every solver
  ├── Seed_Variability   — std-dev of gap per (algorithm, n)
  └── Win_Rates          — how often each algorithm finds the best objective

  Figures  (analysis_output/figures/)
  ├── fig1_gap_vs_n.png         — True gap vs n, faceted by β
  ├── fig2_gap_vs_beta.png      — True gap vs β, faceted by n
  ├── fig3_runtime_scaling.png  — Runtime vs n, log scale, all solvers
  ├── fig4_gap_boxplots.png     — Gap distribution per algorithm
  ├── fig5_gap_heatmap.png      — (n, β) heatmap of mean gap per algorithm
  └── fig6_seed_variability.png — Std-dev of gap per (algorithm, n)

Design principles
-----------------
- Adding new heuristics requires no change to this file.
- All plots use a consistent, publication-ready style (white background,
  colour-blind-friendly palette, 300 dpi, no chart junk).
- Density keys in filenames are parsed as integers (d{int(β*10000):04d})
  to avoid floating-point comparison issues.
- Gurobi optima are optional; every metric degrades gracefully to NaN
  when no optimal file exists for an instance.

Usage
-----
    python analysis.py
    python analysis.py --results-dir results --analysis-dir analysis_output
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import warnings
from typing import Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  STYLE
# ══════════════════════════════════════════════════════════════════════════════

# Colour-blind-friendly palette (up to 8 algorithms)
_PALETTE = [
    "#2166AC",  # blue
    "#D6604D",  # red-orange
    "#4DAC26",  # green
    "#8073AC",  # purple
    "#F4A582",  # salmon
    "#92C5DE",  # light blue
    "#A6D96A",  # light green
    "#F1A340",  # amber
]

_FIG_DPI  = 300
_FONT_SZ  = 11
_TITLE_SZ = 13
_AXIS_SZ  = 10


def _set_style() -> None:
    """Apply a clean, publication-ready matplotlib style globally."""
    plt.rcParams.update({
        "figure.dpi":        _FIG_DPI,
        "savefig.dpi":       _FIG_DPI,
        "savefig.bbox":      "tight",
        "font.size":         _FONT_SZ,
        "axes.titlesize":    _TITLE_SZ,
        "axes.labelsize":    _AXIS_SZ,
        "xtick.labelsize":   _AXIS_SZ,
        "ytick.labelsize":   _AXIS_SZ,
        "legend.fontsize":   9,
        "legend.framealpha": 0.85,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.35,
        "grid.linestyle":    "--",
        "lines.linewidth":   1.8,
        "lines.markersize":  6,
        "font.family":       "sans-serif",
    })
    sns.set_palette(_PALETTE)


def _algo_colors(algos) -> Dict[str, str]:
    """Return a stable {algorithm: colour} mapping."""
    return {a: _PALETTE[i % len(_PALETTE)] for i, a in enumerate(sorted(algos))}


def _save(fig, path: str) -> None:
    fig.savefig(path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _parse_optimal_key(fname: str) -> Optional[tuple]:
    """Extract (n, density_tag_int, seed) from optimal_nX_dY_sZ.json."""
    try:
        parts = os.path.basename(fname).replace(".json", "").split("_")
        return int(parts[1][1:]), int(parts[2][1:]), int(parts[3][1:])
    except Exception:
        return None


def _result_key(data: dict) -> Optional[tuple]:
    """Derive (n, density_tag_int, seed) from a result JSON payload."""
    try:
        n    = int(data["n"])
        seed = int(data["seed"])
        density = (data.get("density")
                   or data.get("conflict_graph_density")
                   or data["num_conflicts"] / (n * n))
        return n, int(round(float(density) * 10000)), seed
    except Exception:
        return None


def load_all_results(results_dir: str = "results") -> pd.DataFrame:
    """Load result_*.json and optimal_*.json into a flat analysis DataFrame."""

    # ── Gurobi optimal solutions ─────────────────────────────────────────────
    optimal:     Dict[tuple, float] = {}
    opt_runtime: Dict[tuple, float] = {}

    for fpath in glob.glob(os.path.join(results_dir, "optimal_*.json")):
        key = _parse_optimal_key(fpath)
        if key is None:
            continue
        try:
            with open(fpath) as f:
                d = json.load(f)
            if d.get("status") in ("OPTIMAL", "TIME_LIMIT") and d.get("objective") is not None:
                optimal[key]     = float(d["objective"])
                opt_runtime[key] = float(d.get("runtime", np.nan))
        except Exception:
            pass

    # ── Heuristic results ────────────────────────────────────────────────────
    records = []

    for fpath in glob.glob(os.path.join(results_dir, "result_*.json")):
        try:
            with open(fpath) as f:
                data = json.load(f)
        except Exception:
            continue

        key = _result_key(data)
        if key is None:
            continue

        n, dtag, seed = key
        beta    = dtag / 10000.0
        opt_obj = optimal.get(key, np.nan)
        opt_rt  = opt_runtime.get(key, np.nan)

        # Subgradient info — tolerate both old and new schema
        sg = data.get("subgradient", {})
        subg_LB  = float(data.get("subgradient_LB")  or sg.get("LB",  np.nan) or np.nan)
        subg_UB  = float(data.get("subgradient_UB")  or sg.get("UB",  np.nan) or np.nan)
        subg_rt  = float(data.get("subgradient_runtime") or sg.get("runtime_seconds", np.nan) or np.nan)
        subg_itr = int(data.get("subgradient_iterations") or sg.get("iterations", 0) or 0)

        heuristic = data.get("heuristic", data.get("heuristic_name", "unknown"))
        heur_out  = data.get("heuristic_output", {})

        def _row(obj, feasible, rt, ordering_label):
            obj = float(obj) if obj is not None else np.nan

            true_gap = (
                max(0.0, 100.0 * (opt_obj - obj) / opt_obj)
                if np.isfinite(obj) and obj > 0 and np.isfinite(opt_obj) and opt_obj > 0
                else np.nan
            )
            dual_gap = (
                max(0.0, 100.0 * (subg_UB - opt_obj) / opt_obj)
                if np.isfinite(subg_UB) and np.isfinite(opt_obj) and opt_obj > 0
                else np.nan
            )
            algo = (f"{heuristic} ({ordering_label})"
                    if ordering_label != "—" else heuristic)
            return {
                "n":               n,
                "beta":            beta,
                "seed":            seed,
                "num_conflicts":   int(data.get("num_conflicts", 0)),
                "heuristic":       heuristic,
                "ordering":        ordering_label,
                "Algorithm":       algo,
                "opt_objective":   opt_obj,
                "opt_runtime_s":   opt_rt,
                "subg_LB":         subg_LB,
                "subg_UB":         subg_UB,
                "subg_runtime_s":  subg_rt,
                "subg_iterations": subg_itr,
                "objective":       obj,
                "feasible":        bool(feasible),
                "heur_runtime_s":  float(rt) if rt is not None else np.nan,
                "true_gap_pct":    true_gap,
                "dual_gap_pct":    dual_gap,
            }

        if "ordering_variants" in heur_out:
            for ord_key, od in heur_out["ordering_variants"].items():
                records.append(_row(
                    obj=od.get("objective"),
                    feasible=od.get("feasible", False),
                    rt=od.get("runtime_seconds"),
                    ordering_label=od.get("ordering_label", ord_key),
                ))
        else:
            records.append(_row(
                obj=heur_out.get("objective"),
                feasible=heur_out.get("feasible", False),
                rt=heur_out.get("runtime_seconds"),
                ordering_label="—",
            ))

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    for col in ("n", "seed", "num_conflicts", "subg_iterations"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ("beta", "opt_objective", "opt_runtime_s", "subg_LB", "subg_UB",
                "subg_runtime_s", "objective", "heur_runtime_s",
                "true_gap_pct", "dual_gap_pct"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["feasible"] = df["feasible"].fillna(False).astype(bool)
    return df.sort_values(["heuristic", "ordering", "n", "beta", "seed"]).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  EXCEL WORKBOOK
# ══════════════════════════════════════════════════════════════════════════════

def _write_sheet(writer, name: str, df: pd.DataFrame) -> None:
    """Write df to a sheet with auto-sized columns."""
    df.to_excel(writer, sheet_name=name[:31], index=False)
    ws = writer.sheets[name[:31]]
    for i, col in enumerate(df.columns, 1):
        w = max(len(str(col)) + 2,
                df[col].astype(str).str.len().max() + 2 if not df.empty else 10)
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = min(w, 40)


def make_excel(df: pd.DataFrame, excel_path: str) -> None:
    feas = df[df["feasible"]].copy()

    # ── Raw Data ──────────────────────────────────────────────────────────────
    raw = df[[
        "n", "beta", "seed", "num_conflicts",
        "heuristic", "ordering", "Algorithm",
        "opt_objective", "opt_runtime_s",
        "subg_LB", "subg_UB", "subg_runtime_s", "subg_iterations",
        "objective", "feasible", "heur_runtime_s",
        "true_gap_pct", "dual_gap_pct",
    ]].round(4)

    # ── Summary by Algorithm ──────────────────────────────────────────────────
    summary = (
        feas.groupby("Algorithm")
        .agg(
            runs               = ("objective",      "count"),
            mean_true_gap_pct  = ("true_gap_pct",   "mean"),
            median_true_gap_pct= ("true_gap_pct",   "median"),
            std_true_gap_pct   = ("true_gap_pct",   "std"),
            mean_dual_gap_pct  = ("dual_gap_pct",   "mean"),
            mean_obj           = ("objective",      "mean"),
            mean_heur_rt_s     = ("heur_runtime_s", "mean"),
        )
        .reset_index()
    )
    # Feasibility computed over all rows (including infeasible)
    summary["feasibility_pct"] = (
        summary["Algorithm"].map(df.groupby("Algorithm")["feasible"].mean() * 100)
    )
    summary = summary.round(3).sort_values("mean_true_gap_pct")

    # ── Gap Table: (n, β) × Algorithm ─────────────────────────────────────────
    gap_pivot = (
        feas.groupby(["n", "beta", "Algorithm"])["true_gap_pct"]
        .mean().unstack("Algorithm").round(2)
    )
    gap_pivot.index.names = ["n", "beta"]
    gap_pivot = gap_pivot.reset_index()

    # ── Runtime Table: n × Solver ─────────────────────────────────────────────
    rt_heur = (
        df.groupby(["n", "Algorithm"])["heur_runtime_s"]
        .mean().unstack("Algorithm").round(4)
    )
    rt_base = (
        df.drop_duplicates(["n", "beta", "seed"])
        .groupby("n")
        .agg(Gurobi_s=("opt_runtime_s", "mean"),
             Subgradient_s=("subg_runtime_s", "mean"))
        .round(3)
    )
    rt_table = rt_base.join(rt_heur).reset_index()

    # ── Seed Variability ──────────────────────────────────────────────────────
    var_table = (
        feas.groupby(["Algorithm", "n"])["true_gap_pct"]
        .agg(mean_gap=("mean"), std_gap=("std"),
             min_gap=("min"),   max_gap=("max"), n_runs=("count"))
        .round(3).reset_index()
    )

    # ── Win Rates ─────────────────────────────────────────────────────────────
    if not feas.empty:
        best = feas.groupby(["n", "beta", "seed"])["objective"].transform("max")
        feas2 = feas.copy()
        feas2["is_best"] = feas2["objective"] >= best - 1e-6
        win = (
            feas2.groupby("Algorithm")["is_best"]
            .agg(wins="sum", runs="count")
            .assign(win_rate_pct=lambda x: (x["wins"] / x["runs"] * 100).round(1))
            .reset_index()
            .sort_values("win_rate_pct", ascending=False)
        )
    else:
        win = pd.DataFrame(columns=["Algorithm", "wins", "runs", "win_rate_pct"])

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        _write_sheet(writer, "Raw_Data",         raw)
        _write_sheet(writer, "Summary_by_Algo",  summary)
        _write_sheet(writer, "Gap_Table",        gap_pivot)
        _write_sheet(writer, "Runtime_Table",    rt_table)
        _write_sheet(writer, "Seed_Variability", var_table)
        _write_sheet(writer, "Win_Rates",        win)

    print(f"  Saved {excel_path}  "
          f"({df['Algorithm'].nunique()} algorithms, {len(df):,} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def fig_gap_vs_n(df: pd.DataFrame, out: str) -> None:
    """Fig 1 — True gap vs n, one panel per β value."""
    feas = df[df["feasible"] & df["true_gap_pct"].notna()]
    if feas.empty:
        return

    betas  = sorted(feas["beta"].unique())
    algos  = sorted(feas["Algorithm"].unique())
    colors = _algo_colors(algos)
    ncols  = min(3, len(betas))
    nrows  = int(np.ceil(len(betas) / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.5 * nrows),
                             sharey=True, squeeze=False)
    fig.suptitle(
        "True Optimality Gap (%) vs Problem Size n\n"
        "averaged over seeds — one panel per conflict density β",
        fontsize=_TITLE_SZ, y=1.02,
    )

    for idx, beta in enumerate(betas):
        ax  = axes[idx // ncols][idx % ncols]
        sub = feas[feas["beta"] == beta]
        grp = sub.groupby(["n", "Algorithm"])["true_gap_pct"].mean().reset_index()

        for algo in algos:
            d = grp[grp["Algorithm"] == algo]
            if d.empty:
                continue
            ax.plot(d["n"], d["true_gap_pct"],
                    marker="o", label=algo, color=colors[algo])

        ax.set_title(f"β = {beta:.3f}", fontsize=_FONT_SZ)
        ax.set_xlabel("n")
        ax.set_ylabel("Mean gap (%)")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    for idx in range(len(betas), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(len(algos), 5), bbox_to_anchor=(0.5, -0.05),
               framealpha=0.9)
    _save(fig, os.path.join(out, "fig1_gap_vs_n.png"))


def fig_gap_vs_beta(df: pd.DataFrame, out: str) -> None:
    """Fig 2 — True gap vs β, one panel per n value."""
    feas = df[df["feasible"] & df["true_gap_pct"].notna()]
    if feas.empty:
        return

    n_vals = sorted(feas["n"].dropna().unique())
    algos  = sorted(feas["Algorithm"].unique())
    colors = _algo_colors(algos)
    ncols  = min(3, len(n_vals))
    nrows  = int(np.ceil(len(n_vals) / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.5 * nrows),
                             sharey=True, squeeze=False)
    fig.suptitle(
        "True Optimality Gap (%) vs Conflict Density β\n"
        "averaged over seeds — one panel per problem size n",
        fontsize=_TITLE_SZ, y=1.02,
    )

    for idx, n in enumerate(n_vals):
        ax  = axes[idx // ncols][idx % ncols]
        sub = feas[feas["n"] == n]
        grp = sub.groupby(["beta", "Algorithm"])["true_gap_pct"].mean().reset_index()

        for algo in algos:
            d = grp[grp["Algorithm"] == algo]
            if d.empty:
                continue
            ax.plot(d["beta"], d["true_gap_pct"],
                    marker="o", label=algo, color=colors[algo])

        ax.set_title(f"n = {n}", fontsize=_FONT_SZ)
        ax.set_xlabel("β")
        ax.set_ylabel("Mean gap (%)")

    for idx in range(len(n_vals), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(len(algos), 5), bbox_to_anchor=(0.5, -0.05),
               framealpha=0.9)
    _save(fig, os.path.join(out, "fig2_gap_vs_beta.png"))


def fig_runtime_scaling(df: pd.DataFrame, out: str) -> None:
    """Fig 3 — Runtime vs n (log scale), all algorithms + Gurobi + subgradient."""
    base   = df.drop_duplicates(["n", "beta", "seed"])
    rt_ref = base.groupby("n").agg(
        gurobi_s      = ("opt_runtime_s",  "mean"),
        subgradient_s = ("subg_runtime_s", "mean"),
    ).reset_index()

    algos  = sorted(df["Algorithm"].unique())
    colors = _algo_colors(algos)

    fig, ax = plt.subplots(figsize=(10, 6))

    if rt_ref["gurobi_s"].notna().any():
        ax.plot(rt_ref["n"], rt_ref["gurobi_s"],
                color="black", linestyle="--", marker="s",
                linewidth=2.2, label="Gurobi (exact)", zorder=5)

    ax.plot(rt_ref["n"], rt_ref["subgradient_s"],
            color="#555555", linestyle=":", marker="d",
            linewidth=2.2, label="Subgradient (dual bound)", zorder=4)

    for algo in algos:
        rt = df[df["Algorithm"] == algo].groupby("n")["heur_runtime_s"].mean().reset_index()
        ax.plot(rt["n"], rt["heur_runtime_s"],
                marker="o", label=algo, color=colors[algo])

    ax.set_yscale("log")
    ax.set_xlabel("Problem size n")
    ax.set_ylabel("Mean runtime (seconds, log scale)")
    ax.set_title("Runtime Scaling: All Solvers vs Problem Size n")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _save(fig, os.path.join(out, "fig3_runtime_scaling.png"))


def fig_gap_boxplots(df: pd.DataFrame, out: str) -> None:
    """Fig 4 — Gap distribution boxplots, one box per algorithm."""
    feas = df[df["feasible"] & df["true_gap_pct"].notna()]
    if feas.empty:
        return

    algos   = sorted(feas["Algorithm"].unique())
    colors  = _algo_colors(algos)
    palette = {a: colors[a] for a in algos}

    fig, ax = plt.subplots(figsize=(10, max(5, len(algos) * 0.75)))
    sns.boxplot(
        data=feas, y="Algorithm", x="true_gap_pct",
        palette=palette, order=algos,
        flierprops=dict(marker=".", markersize=3, alpha=0.4),
        ax=ax,
    )
    # Diamond = mean
    means = feas.groupby("Algorithm")["true_gap_pct"].mean()
    for i, algo in enumerate(algos):
        if algo in means.index:
            ax.plot(means[algo], i, marker="D", color="white",
                    markeredgecolor="black", markersize=6, zorder=5)

    ax.axvline(0, color="gray", linewidth=0.9, linestyle="--")
    ax.set_xlabel("True gap from Gurobi optimum (%)")
    ax.set_ylabel("")
    ax.set_title(
        "Distribution of True Optimality Gap by Algorithm\n"
        "(all n, all β, all seeds  |  ◆ = mean)"
    )
    _save(fig, os.path.join(out, "fig4_gap_boxplots.png"))


def fig_gap_heatmap(df: pd.DataFrame, out: str) -> None:
    """Fig 5 — (n, β) heatmap of mean gap, one subplot per algorithm."""
    feas = df[df["feasible"] & df["true_gap_pct"].notna()]
    if feas.empty:
        return

    algos = sorted(feas["Algorithm"].unique())
    ncols = min(2, len(algos))
    nrows = int(np.ceil(len(algos) / ncols))
    vmax  = feas["true_gap_pct"].quantile(0.95)  # cap colour scale

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7.5 * ncols, 5.5 * nrows),
                             squeeze=False)
    fig.suptitle(
        "Mean True Gap (%) — (n, β) Heat Map per Algorithm",
        fontsize=_TITLE_SZ, y=1.02,
    )

    for idx, algo in enumerate(algos):
        ax    = axes[idx // ncols][idx % ncols]
        sub   = feas[feas["Algorithm"] == algo]
        pivot = (
            sub.groupby(["n", "beta"])["true_gap_pct"]
            .mean().unstack("beta")
        )
        sns.heatmap(
            pivot, ax=ax,
            cmap="YlOrRd", annot=True, fmt=".1f",
            vmin=0, vmax=vmax,
            linewidths=0.4, linecolor="white",
            cbar_kws={"label": "Mean gap (%)"},
        )
        ax.set_title(algo, fontsize=_FONT_SZ)
        ax.set_xlabel("Conflict density β")
        ax.set_ylabel("Problem size n")

    for idx in range(len(algos), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    _save(fig, os.path.join(out, "fig5_gap_heatmap.png"))


def fig_seed_variability(df: pd.DataFrame, out: str) -> None:
    """Fig 6 — Std-dev of gap across seeds: measures robustness per algorithm."""
    feas = df[df["feasible"] & df["true_gap_pct"].notna()]
    if feas.empty:
        return

    algos  = sorted(feas["Algorithm"].unique())
    colors = _algo_colors(algos)

    var = (
        feas.groupby(["Algorithm", "n"])["true_gap_pct"]
        .std().reset_index()
        .rename(columns={"true_gap_pct": "std_gap_pct"})
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    for algo in algos:
        d = var[var["Algorithm"] == algo]
        ax.plot(d["n"], d["std_gap_pct"],
                marker="o", label=algo, color=colors[algo])

    ax.set_xlabel("Problem size n")
    ax.set_ylabel("Std dev of true gap (%) across seeds")
    ax.set_title(
        "Seed Variability — Robustness of Each Algorithm\n"
        "(lower = more consistent across random instances)"
    )
    ax.legend(loc="upper left", framealpha=0.9)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _save(fig, os.path.join(out, "fig6_seed_variability.png"))


# ══════════════════════════════════════════════════════════════════════════════
# 5.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def clean_outputs(analysis_dir: str, figures_dir: str) -> None:
    os.makedirs(figures_dir, exist_ok=True)
    for pattern in (os.path.join(analysis_dir, "*.xlsx"),
                    os.path.join(figures_dir,  "*.png")):
        for f in glob.glob(pattern):
            os.remove(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Academic-grade analysis for MAX-APC experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir",  default="results",
                        help="Directory with result_*.json and optimal_*.json files.")
    parser.add_argument("--analysis-dir", default="analysis_output",
                        help="Root directory for all outputs.")
    args = parser.parse_args()

    figures_dir = os.path.join(args.analysis_dir, "figures")

    print("── Cleaning old outputs ──────────────────────────────────────────")
    clean_outputs(args.analysis_dir, figures_dir)

    print("── Loading results ───────────────────────────────────────────────")
    _set_style()
    df = load_all_results(args.results_dir)

    if df.empty:
        print("No results found. Run batch_experiment.py (and optionally "
              "gurobi_batch.py) first.")
        return

    has_opt = df["opt_objective"].notna().any()
    print(f"  {len(df):,} rows | {df['Algorithm'].nunique()} algorithms | "
          f"{df[['n','beta','seed']].drop_duplicates().shape[0]} unique instances | "
          f"Gurobi optima: {'yes' if has_opt else 'no — gap metrics will be NaN'}")

    print("\n── Writing Excel workbook ────────────────────────────────────────")
    make_excel(df, os.path.join(args.analysis_dir, "analysis_tables.xlsx"))

    print("\n── Generating figures ────────────────────────────────────────────")
    fig_gap_vs_n(df, figures_dir)
    fig_gap_vs_beta(df, figures_dir)
    fig_runtime_scaling(df, figures_dir)
    fig_gap_boxplots(df, figures_dir)
    fig_gap_heatmap(df, figures_dir)
    fig_seed_variability(df, figures_dir)

    print(f"\n── Done.  All outputs in: {args.analysis_dir}/ ─────────────────")


if __name__ == "__main__":
    main()
