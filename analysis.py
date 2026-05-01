"""
analysis.py
============

Academic-grade analysis pipeline for MAX-APC experiments.

Ingests raw JSON outputs from `results/` (heuristic runs and Gurobi optima)
and produces statistical tables (CSV + LaTeX + Excel) and publication-ready
figures (300 dpi PNG) under `analysis_output/`.

Phase 1   — Data ingestion + null handling for academic integrity.
Phase 2   — Tables: feasibility matrix, runtime pivot, bounds/gap, win rates,
            Wilcoxon signed-rank pairwise tests.
Phase 3   — Figures: convergence dynamics, runtime scaling, gap boxplots,
            phase-transition heatmap (Goldilocks zone), Dolan-Moré profile.

Re-running this script overwrites everything under analysis_output/.

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
import re
import shutil
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# 1. CONSTANTS, STYLE, AND COLOUR PALETTE
# =============================================================================

GUROBI_LABEL = "Gurobi"

# Colorblind-friendly palette (Seaborn 'colorblind' + neutral for Gurobi).
# Specific algorithms get fixed colours so they appear identical across every
# figure in the thesis. Add new heuristic names as they appear.
_ALGO_COLORS: Dict[str, str] = {
    GUROBI_LABEL:                "#000000",   # black, dashed in plots
    "lagrangean_repair":          "#0173B2",   # cb-blue
    "lagrangean_repair_2":        "#DE8F05",   # cb-orange
    "lagrangean_repair_lambda":   "#029E73",   # cb-green
    "lagrangean_repair_savlr":    "#D55E00",   # cb-vermillion
    # Stable fallback colours for any new heuristics that show up.
    "_fallback_pool": [
        "#CC78BC", "#CA9161", "#FBAFE4", "#949494", "#ECE133", "#56B4E9",
    ],
}

_INSTANCE_CATEGORY_ORDER = ["standard", "goldilocks", "degen", "extreme"]
_CATEGORY_DISPLAY = {
    "standard":   "Standard",
    "goldilocks": "Goldilocks",
    "degen":      "Degeneracy",
    "extreme":    "Extreme",
}

_FIG_DPI = 300


def _color_for(algo: str) -> str:
    """Return a stable colour for a given algorithm name."""
    if algo in _ALGO_COLORS:
        return _ALGO_COLORS[algo]
    pool = _ALGO_COLORS["_fallback_pool"]
    # Hash the name into the pool so order doesn't matter.
    idx = abs(hash(algo)) % len(pool)
    return pool[idx]


def _set_style() -> None:
    """Apply a clean, publication-ready matplotlib style globally."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "figure.dpi":        _FIG_DPI,
        "savefig.dpi":       _FIG_DPI,
        "savefig.bbox":      "tight",
        "font.size":         11,
        "axes.titlesize":    12,
        "axes.labelsize":    10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "legend.framealpha": 0.85,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
        "lines.linewidth":   1.6,
        "lines.markersize":  5.5,
        "font.family":       "sans-serif",
    })


def _save(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# =============================================================================
# 2. PHASE 1 — DATA INGESTION
# =============================================================================

# Filename grammar (see apc_base._result_filename / _instance_filename):
#   standard heuristic result:    {heur}_n{n}_a{aa}_b{bbb}_s{seed}.json
#   difficult heuristic result:   difficult_{heur}_{cat}_n{n}_a{aa}_b{bbb}_s{seed}.json
#   Gurobi optimal (standard):    optimal_n{n}_a{aa}_b{bbb}_s{seed}.json
#   Gurobi optimal (difficult):   difficult_optimal_{cat}_n{n}_a{aa}_b{bbb}_s{seed}.json
# alpha tag = round(α*10):02d   (0.4 → "04",  1.0 → "10")
# beta  tag = round(β*1000):03d (0.01 → "010", 0.001 → "001")

_SUFFIX_RE = re.compile(r"_n(\d+)_a(\d+)_b(\d+)_s(\d+)$")
_DIFFICULT_CATEGORIES = ("goldilocks", "degen", "extreme")


def _parse_result_filename(fname: str) -> Optional[Dict]:
    """Decode (algo, category, n, alpha, beta, seed) from a JSON filename.

    Returns None for files that aren't a result/optimal artefact (e.g. bare
    instance files in a mixed directory, or partials we don't recognise).
    """
    base = os.path.basename(fname)
    if not base.endswith(".json"):
        return None
    stem = base[:-5]

    m = _SUFFIX_RE.search(stem)
    if not m:
        return None
    n, alpha_tag, beta_tag, seed = (int(g) for g in m.groups())
    head = stem[: m.start()]

    # head is one of:
    #   "instance"             → not a result, skip
    #   "optimal"              → Gurobi standard
    #   "{heur}"               → heuristic standard
    #   "difficult_instance_{cat}" → not a result, skip (it's an instance file)
    #   "difficult_optimal_{cat}"  → Gurobi difficult
    #   "difficult_{heur}_{cat}"   → heuristic difficult
    if head == "instance":
        return None
    if head.startswith("difficult_instance_"):
        return None

    if head == "optimal":
        return {"algo": GUROBI_LABEL, "kind": "optimal", "category": "standard",
                "n": n, "alpha": alpha_tag / 10.0, "beta": beta_tag / 1000.0,
                "seed": seed}

    if head.startswith("difficult_"):
        rest = head[len("difficult_"):]
        # Difficult Gurobi: "optimal_{cat}"
        if rest.startswith("optimal_"):
            cat = rest[len("optimal_"):]
            if cat not in _DIFFICULT_CATEGORIES:
                return None
            return {"algo": GUROBI_LABEL, "kind": "optimal", "category": cat,
                    "n": n, "alpha": alpha_tag / 10.0, "beta": beta_tag / 1000.0,
                    "seed": seed}
        # Difficult heuristic: "{heur}_{cat}"
        for cat in _DIFFICULT_CATEGORIES:
            suffix = "_" + cat
            if rest.endswith(suffix):
                heur = rest[: -len(suffix)]
                return {"algo": heur, "kind": "heuristic", "category": cat,
                        "n": n, "alpha": alpha_tag / 10.0, "beta": beta_tag / 1000.0,
                        "seed": seed}
        return None

    # Plain heuristic on standard instance.
    return {"algo": head, "kind": "heuristic", "category": "standard",
            "n": n, "alpha": alpha_tag / 10.0, "beta": beta_tag / 1000.0,
            "seed": seed}


def _load_json(fpath: str) -> Optional[Dict]:
    try:
        with open(fpath, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"  ! Could not parse {fpath}: {e}", file=sys.stderr)
        return None


def _heur_objective(data: Dict) -> Tuple[Optional[float], bool]:
    """Best feasible objective from a heuristic JSON, plus the failure flag.

    Implements the academic-integrity rule:
      - If `feasible_found` is explicitly False → NaN (failure).
      - If `incumbent_objective` is None → NaN (failure).
      - If `incumbent_objective` == 0.0 AND E0_objective == 0 → NaN
        (the trivial E0 fallback).
    Returns (objective_or_nan, ran_successfully).
    """
    if data.get("feasible_found") is False:
        return None, False

    obj = data.get("incumbent_objective")
    if obj is None:
        return None, False

    obj = float(obj)
    e0 = float(data.get("E0_objective", 0.0) or 0.0)
    # E0 has cost 0 by construction; treat a 0.0 incumbent as the trivial fallback.
    if abs(obj) < 1e-12 and abs(e0) < 1e-12:
        return None, False
    return obj, True


def _heur_runtime_total(data: Dict) -> Optional[float]:
    """Total wall-clock cost: subgradient ascent + repair heuristic stage."""
    sg = data.get("subgradient_runtime")
    h_out = data.get("heuristic_output", {}) or {}
    hr = h_out.get("runtime_seconds")

    parts = [v for v in (sg, hr) if v is not None]
    if not parts:
        return None
    return float(sum(parts))


def _gurobi_objective(data: Dict) -> Tuple[Optional[float], str, Optional[float], int, int]:
    """Best objective Gurobi found, status, gap, nodes explored, solutions found.

    nodes_explored  — B&B tree size; high values signal a weak LP relaxation.
    solutions_found — 0 means timeout-without-incumbent (hardest case).
    Both fields are absent in legacy JSONs and fall back to NaN / 0.
    """
    status = data.get("status", "UNKNOWN")
    obj = data.get("objective")
    gap = data.get("gap")
    nodes = data.get("nodes_explored")
    sols  = data.get("solutions_found")
    obj_out = float(obj) if obj is not None else None
    gap_out = float(gap) if gap is not None else None
    nodes_out = int(nodes) if nodes is not None else np.nan
    sols_out  = int(sols)  if sols  is not None else np.nan
    return obj_out, status, gap_out, nodes_out, sols_out


def load_master(results_dir: str = "results") -> pd.DataFrame:
    """Load every result_*.json + optimal_*.json into a single long-format frame.

    One row per (algo, category, n, alpha, beta, seed). Failures → NaN
    objective so downstream `.mean()` / `.std()` skip them naturally.
    """
    heuristic_records: List[Dict] = []
    optimal_records: List[Dict] = []

    files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    for fpath in files:
        meta = _parse_result_filename(fpath)
        if meta is None:
            continue
        data = _load_json(fpath)
        if data is None:
            continue

        if meta["kind"] == "optimal":
            obj, status, gap, nodes, sols = _gurobi_objective(data)
            optimal_records.append({
                **meta,
                "objective":           obj,
                "status":              status,
                "runtime_total":       float(data.get("runtime", np.nan)) if data.get("runtime") is not None else np.nan,
                "gap_solver":          gap,
                "feasible":            obj is not None,
                # Bounds: for Gurobi we treat the optimum (or best incumbent) as both LB and UB.
                "LB":                  obj,
                "UB":                  obj,
                "iter_count":          np.nan,
                "first_feasible_time": np.nan,
                "nodes_explored":      nodes,
                "solutions_found":     sols,
            })
            continue

        # Heuristic record.
        obj, ran = _heur_objective(data)
        runtime = _heur_runtime_total(data)
        ub_val = data.get("subgradient_UB")
        ub_val = float(ub_val) if ub_val is not None else np.nan
        # First time the LB strictly improved beyond E0 (= time-to-feasible).
        first_t = _first_feasible_time(data)

        heuristic_records.append({
            **meta,
            "objective":          obj,
            "status":             "FEASIBLE" if ran else "NO_FEASIBLE",
            "runtime_total":      runtime if runtime is not None else np.nan,
            "gap_solver":         np.nan,
            "feasible":           ran,
            "LB":                 obj,
            "UB":                 ub_val,
            "iter_count":         int(data.get("subgradient_iterations", 0) or 0),
            "first_feasible_time": first_t,
            "terminated_reason":  data.get("subgradient_terminated_reason"),
            # Gurobi-specific fields — NaN for heuristic rows.
            "nodes_explored":     np.nan,
            "solutions_found":    np.nan,
        })

    df = pd.DataFrame(heuristic_records + optimal_records)

    if df.empty:
        return df

    # Type coercion.
    for col in ("n", "seed", "iter_count", "nodes_explored", "solutions_found"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ("alpha", "beta", "objective", "runtime_total", "gap_solver",
                "LB", "UB", "first_feasible_time"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["feasible"] = df["feasible"].fillna(False).astype(bool)

    # Stable instance key (used everywhere).
    df["instance_key"] = (
        df["category"] + "|n" + df["n"].astype(str) +
        "|a" + df["alpha"].round(2).astype(str) +
        "|b" + df["beta"].round(4).astype(str) +
        "|s" + df["seed"].astype(str)
    )

    # Friendly category label for plot/table titles.
    df["category_label"] = df["category"].map(_CATEGORY_DISPLAY).fillna(df["category"])

    return df.sort_values(["category", "algo", "n", "alpha", "beta", "seed"]).reset_index(drop=True)


def _first_feasible_time(data: Dict) -> float:
    """Wall-clock seconds at which LB first improved beyond E0 baseline."""
    history = data.get("subgradient_history") or []
    if not history:
        return float("nan")
    e0 = float(data.get("E0_objective", 0.0) or 0.0)
    for rec in history:
        lb = rec.get("LB")
        if lb is None:
            continue
        if float(lb) > e0 + 1e-9:
            return float(rec.get("elapsed_s", "nan"))
    return float("nan")


# =============================================================================
# 3. EXPORT HELPERS  (CSV + LaTeX + Excel sheet)
# =============================================================================

def _export_table(df: pd.DataFrame, name: str, tables_dir: str,
                  caption: str = "", excel_writer=None,
                  index: bool = False, float_fmt: str = "%.3f") -> None:
    """Persist a DataFrame as CSV, LaTeX, and an Excel sheet."""
    if df is None or df.empty:
        print(f"  - skipped {name}: empty")
        return

    csv_path = os.path.join(tables_dir, f"{name}.csv")
    df.to_csv(csv_path, index=index, float_format=float_fmt)

    tex_path = os.path.join(tables_dir, f"{name}.tex")
    try:
        latex = df.to_latex(
            index=index,
            float_format=lambda v: ("" if pd.isna(v) else f"{v:.3f}"),
            na_rep="--",
            caption=caption or name.replace("_", " ").title(),
            label=f"tab:{name}",
            escape=True,
        )
        with open(tex_path, "w") as f:
            f.write(latex)
    except Exception as e:
        print(f"  ! LaTeX export of {name} failed: {e}", file=sys.stderr)

    if excel_writer is not None:
        sheet = name[:31]  # Excel sheet name length cap
        df.to_excel(excel_writer, sheet_name=sheet, index=index)

    print(f"  Saved {csv_path}  +  {os.path.basename(tex_path)}")


# =============================================================================
# 4. PHASE 2 — STATISTICAL TABLES
# =============================================================================

def table_feasibility_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Categories × Algorithms → success rate (%).

    Success here means the run produced a real feasible incumbent (NOT the
    E0 fallback for heuristics, NOT a Gurobi timeout-without-incumbent).
    """
    g = (
        df.groupby(["category", "algo"])
          .agg(success_rate=("feasible", "mean"),
               runs=("feasible", "size"))
          .reset_index()
    )
    g["success_rate"] = (g["success_rate"] * 100.0).round(2)

    pivot = g.pivot(index="category", columns="algo", values="success_rate")
    pivot.index.name = "Category"
    # Order rows
    order = [c for c in _INSTANCE_CATEGORY_ORDER if c in pivot.index]
    pivot = pivot.loc[order]
    pivot.index = [_CATEGORY_DISPLAY.get(c, c) for c in pivot.index]
    return pivot.reset_index()


def table_runtime_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """(n, α, β) × Algorithm → mean ± std runtime (s), averaged over seeds."""
    sub = df.dropna(subset=["runtime_total"]).copy()
    if sub.empty:
        return pd.DataFrame()

    agg = (
        sub.groupby(["category", "n", "alpha", "beta", "algo"])
           .agg(mean_runtime=("runtime_total", "mean"),
                std_runtime=("runtime_total", "std"),
                seeds=("runtime_total", "size"))
           .reset_index()
    )
    # Wide form: one column per algo for the mean, plus a second for std.
    mean_pv = agg.pivot_table(index=["category", "n", "alpha", "beta"],
                              columns="algo", values="mean_runtime")
    std_pv = agg.pivot_table(index=["category", "n", "alpha", "beta"],
                             columns="algo", values="std_runtime")
    std_pv.columns = [f"{c}_std" for c in std_pv.columns]
    out = pd.concat([mean_pv, std_pv], axis=1).reset_index()
    out["category"] = out["category"].map(_CATEGORY_DISPLAY).fillna(out["category"])
    return out.round(4)


def table_bounds_gap(df: pd.DataFrame) -> pd.DataFrame:
    """(n, α, β) × Heuristic → mean LB, mean UB, true gap %, relative gap %.

    True gap     = ((Gurobi_opt − Heuristic_LB) / Gurobi_opt) × 100, where
                   Gurobi_opt is the optimum (or best incumbent) on that
                   instance.
    Relative gap = ((Best_Heur_LB − Heuristic_LB) / Best_Heur_LB) × 100,
                   used when no Gurobi optimum exists on that instance
                   (Goldilocks/Extreme).
    """
    if df.empty:
        return pd.DataFrame()

    # 1. Best Gurobi objective per instance (NaN if unavailable).
    g = df[df["algo"] == GUROBI_LABEL].set_index("instance_key")["objective"]
    gurobi_obj = g.to_dict()

    # 2. Best heuristic LB per instance — used as the relative-gap reference.
    heur_only = df[df["algo"] != GUROBI_LABEL]
    best_heur = (
        heur_only.dropna(subset=["LB"])
                 .groupby("instance_key")["LB"].max()
                 .to_dict()
    )

    rows = []
    for _, r in heur_only.iterrows():
        ikey = r["instance_key"]
        lb = r["LB"]
        ub = r["UB"]
        opt = gurobi_obj.get(ikey)
        ref = best_heur.get(ikey)

        if pd.notna(lb) and opt is not None and not pd.isna(opt) and opt > 0:
            true_gap = max(0.0, 100.0 * (opt - lb) / opt)
        else:
            true_gap = np.nan

        if pd.notna(lb) and ref is not None and not pd.isna(ref) and ref > 0:
            rel_gap = max(0.0, 100.0 * (ref - lb) / ref)
        else:
            rel_gap = np.nan

        rows.append({
            "category":  r["category"],
            "n":         int(r["n"]) if pd.notna(r["n"]) else np.nan,
            "alpha":     r["alpha"],
            "beta":      r["beta"],
            "algo":      r["algo"],
            "LB":        lb,
            "UB":        ub,
            "true_gap_pct": true_gap,
            "relative_gap_pct": rel_gap,
        })

    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw

    agg = (
        raw.groupby(["category", "n", "alpha", "beta", "algo"])
           .agg(mean_LB=("LB", "mean"),
                mean_UB=("UB", "mean"),
                mean_true_gap_pct=("true_gap_pct", "mean"),
                mean_relative_gap_pct=("relative_gap_pct", "mean"),
                runs=("LB", "size"),
                feasible_runs=("LB", "count"))
           .reset_index()
    )
    agg["category"] = agg["category"].map(_CATEGORY_DISPLAY).fillna(agg["category"])
    return agg.round(3)


def table_win_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Per-algorithm win counts on (LB, time-to-feasible) across all instances."""
    if df.empty:
        return pd.DataFrame()
    heur = df[df["algo"] != GUROBI_LABEL].copy()
    if heur.empty:
        return pd.DataFrame()

    n_instances = heur["instance_key"].nunique()

    # Best LB winner: argmax over algos with a real LB (NaNs excluded).
    feas = heur.dropna(subset=["LB"])
    if not feas.empty:
        winners_lb = (
            feas.loc[feas.groupby("instance_key")["LB"].idxmax(), "algo"]
                .value_counts()
        )
    else:
        winners_lb = pd.Series(dtype=int)

    # Fastest-to-feasible: argmin over first_feasible_time.
    fast = heur.dropna(subset=["first_feasible_time"])
    if not fast.empty:
        winners_t = (
            fast.loc[fast.groupby("instance_key")["first_feasible_time"].idxmin(), "algo"]
                .value_counts()
        )
    else:
        winners_t = pd.Series(dtype=int)

    algos = sorted(heur["algo"].unique())
    rows = []
    for a in algos:
        wins_lb = int(winners_lb.get(a, 0))
        wins_t = int(winners_t.get(a, 0))
        rows.append({
            "Algorithm":            a,
            "Best_LB_wins":         wins_lb,
            "Best_LB_pct":          round(100.0 * wins_lb / max(n_instances, 1), 2),
            "Fastest_feasible_wins": wins_t,
            "Fastest_feasible_pct": round(100.0 * wins_t / max(n_instances, 1), 2),
            "instances":            n_instances,
        })
    return pd.DataFrame(rows).sort_values("Best_LB_wins", ascending=False).reset_index(drop=True)


def table_wilcoxon_pairwise(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank tests on true optimality gaps.

    Tests, for each (algo_A, algo_B) pair where both produced a feasible
    solution on the same instance: H0 = "gaps are paired-equal".
    """
    if df.empty:
        return pd.DataFrame()
    heur = df[df["algo"] != GUROBI_LABEL].copy()
    g_obj = (
        df[df["algo"] == GUROBI_LABEL]
        .set_index("instance_key")["objective"]
        .to_dict()
    )

    # Build per-(algo, instance) gap series.
    heur = heur.dropna(subset=["LB"])
    heur["true_gap_pct"] = heur.apply(
        lambda r: (100.0 * (g_obj[r["instance_key"]] - r["LB"]) / g_obj[r["instance_key"]])
                  if (r["instance_key"] in g_obj
                      and g_obj[r["instance_key"]] is not None
                      and not pd.isna(g_obj[r["instance_key"]])
                      and g_obj[r["instance_key"]] > 0)
                  else np.nan,
        axis=1,
    )
    heur = heur.dropna(subset=["true_gap_pct"])
    if heur.empty:
        return pd.DataFrame()

    pivot = heur.pivot_table(index="instance_key", columns="algo",
                             values="true_gap_pct", aggfunc="mean")
    algos = list(pivot.columns)

    rows = []
    for i, a in enumerate(algos):
        for b in algos[i + 1:]:
            paired = pivot[[a, b]].dropna()
            if len(paired) < 6:
                # Wilcoxon needs at least a handful of pairs to be meaningful.
                rows.append({"algo_A": a, "algo_B": b, "n_pairs": len(paired),
                             "median_diff_A_minus_B": np.nan,
                             "statistic": np.nan, "p_value": np.nan,
                             "significant_at_0.05": False})
                continue
            try:
                stat, p = stats.wilcoxon(paired[a].values, paired[b].values,
                                         zero_method="wilcox", alternative="two-sided")
            except Exception:
                stat, p = np.nan, np.nan
            rows.append({
                "algo_A":               a,
                "algo_B":               b,
                "n_pairs":              int(len(paired)),
                "median_diff_A_minus_B": float(np.median(paired[a] - paired[b])),
                "statistic":            float(stat) if stat is not None else np.nan,
                "p_value":              float(p) if p is not None else np.nan,
                "significant_at_0.05":  bool(p is not None and not np.isnan(p) and p < 0.05),
            })
    return pd.DataFrame(rows).round(5)


def table_gurobi_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """Per-instance Gurobi difficulty metrics: nodes explored, solutions found, gap.

    Rows where nodes_explored is NaN are legacy results that pre-date the
    nodes_explored / solutions_found fields in gurobi_solver.py — they are
    included with those columns blank so the objective / gap columns still
    contribute to the table.

    Useful for the thesis narrative:
      - High nodes_explored + high gap  → weak LP relaxation (e.g. a10_b100).
      - solutions_found == 0            → Gurobi timed out before any incumbent.
    """
    gurobi = df[df["algo"] == GUROBI_LABEL].copy()
    if gurobi.empty:
        return pd.DataFrame()
    cols = ["category", "n", "alpha", "beta", "seed",
            "status", "objective", "gap_solver",
            "nodes_explored", "solutions_found", "runtime_total"]
    out = gurobi[[c for c in cols if c in gurobi.columns]].copy()
    out["category"] = out["category"].map(_CATEGORY_DISPLAY).fillna(out["category"])
    out = out.sort_values(["category", "n", "alpha", "beta", "seed"]).reset_index(drop=True)
    for col in ("objective", "gap_solver", "runtime_total"):
        if col in out.columns:
            out[col] = out[col].round(4)
    return out


# =============================================================================
# 5. PHASE 3 — FIGURES
# =============================================================================

def fig_convergence_dynamics(df: pd.DataFrame, results_dir: str,
                             figures_dir: str, max_panels: int = 4) -> None:
    """LB and UB trajectories vs elapsed_s for representative hard instances.

    Picks up to `max_panels` (n, α, β) cells from the goldilocks + extreme
    categories, averages the subgradient_history per algorithm across seeds,
    and plots LB (solid) / UB (dashed) on a single panel per cell.
    """
    if df.empty:
        return

    candidates = (
        df[df["category"].isin(["goldilocks", "extreme", "degen"])]
          [["category", "n", "alpha", "beta", "instance_key"]]
          .drop_duplicates()
    )
    if candidates.empty:
        candidates = df[["category", "n", "alpha", "beta", "instance_key"]].drop_duplicates()
    if candidates.empty:
        return

    # Select the largest-n cells in each category.
    cells = (
        candidates.drop_duplicates(subset=["category", "n", "alpha", "beta"])
                  .sort_values(["category", "n"], ascending=[True, False])
                  .head(max_panels)
                  .reset_index(drop=True)
    )

    # Pre-load all the heuristic JSONs we need (avoid re-globbing per panel).
    json_cache: Dict[str, Dict] = {}
    for fpath in glob.glob(os.path.join(results_dir, "*.json")):
        meta = _parse_result_filename(fpath)
        if meta is None or meta["kind"] != "heuristic":
            continue
        json_cache[fpath] = {**meta, "data": _load_json(fpath)}

    n_panels = len(cells)
    ncols = min(2, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows),
                             squeeze=False)

    for idx, cell in cells.iterrows():
        ax = axes[idx // ncols][idx % ncols]
        cell_jsons = [
            v for v in json_cache.values()
            if v["category"] == cell["category"]
               and v["n"] == cell["n"]
               and abs(v["alpha"] - cell["alpha"]) < 1e-9
               and abs(v["beta"] - cell["beta"]) < 1e-9
        ]
        if not cell_jsons:
            ax.set_visible(False)
            continue

        for algo in sorted({v["algo"] for v in cell_jsons}):
            same_algo = [v["data"] for v in cell_jsons if v["algo"] == algo and v["data"]]
            histories = [d.get("subgradient_history") or [] for d in same_algo]
            histories = [h for h in histories if h]
            if not histories:
                continue

            # Average LB/UB across seeds onto a common time grid (longest run).
            longest = max(histories, key=len)
            t_grid = np.array([rec.get("elapsed_s", np.nan) for rec in longest])
            lb_stack, ub_stack = [], []
            for h in histories:
                lbs = np.array([rec.get("LB", np.nan) for rec in h])
                ubs = np.array([rec.get("UB", np.nan) for rec in h])
                # Pad / truncate to t_grid length for averaging.
                if len(lbs) < len(t_grid):
                    pad = np.full(len(t_grid) - len(lbs), np.nan)
                    lbs = np.concatenate([lbs, pad])
                    ubs = np.concatenate([ubs, pad])
                lb_stack.append(lbs[: len(t_grid)])
                ub_stack.append(ubs[: len(t_grid)])
            lb_mean = np.nanmean(np.vstack(lb_stack), axis=0)
            ub_mean = np.nanmean(np.vstack(ub_stack), axis=0)

            color = _color_for(algo)
            ax.plot(t_grid, lb_mean, "-", color=color, label=f"{algo} LB")
            ax.plot(t_grid, ub_mean, "--", color=color, alpha=0.7, label=f"{algo} UB")

        ax.set_title(
            f"{_CATEGORY_DISPLAY.get(cell['category'], cell['category'])}  "
            f"n={cell['n']}, α={cell['alpha']:.2f}, β={cell['beta']:.3f}",
        )
        ax.set_xlabel("Elapsed time (s)")
        ax.set_ylabel("Bound value")
        ax.legend(fontsize=7, loc="lower right")

    for j in range(n_panels, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle("Convergence Dynamics: LB / UB vs Time (averaged over seeds)",
                 y=1.02)
    _save(fig, os.path.join(figures_dir, "fig_convergence_dynamics.png"))


def fig_runtime_scaling(df: pd.DataFrame, figures_dir: str) -> None:
    """Mean runtime (log y) vs n, one line per algorithm, all categories."""
    sub = df.dropna(subset=["runtime_total"]).copy()
    if sub.empty:
        return
    agg = sub.groupby(["algo", "n"])["runtime_total"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    for algo in sorted(agg["algo"].unique()):
        d = agg[agg["algo"] == algo].sort_values("n")
        if d.empty:
            continue
        style = "--" if algo == GUROBI_LABEL else "-"
        ax.plot(d["n"], d["runtime_total"], marker="o",
                color=_color_for(algo), linestyle=style, label=algo,
                linewidth=2.0 if algo == GUROBI_LABEL else 1.6)

    ax.set_yscale("log")
    ax.set_xlabel("Problem size n")
    ax.set_ylabel("Mean runtime (s, log scale)")
    ax.set_title("Asymptotic Scalability — CPU Runtime vs Problem Size n")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(loc="upper left", framealpha=0.9)
    _save(fig, os.path.join(figures_dir, "fig_runtime_scaling.png"))


def fig_gap_boxplots(df: pd.DataFrame, figures_dir: str) -> None:
    """Boxplot of true optimality gap (%) by category × heuristic."""
    if df.empty:
        return
    g_obj = (
        df[df["algo"] == GUROBI_LABEL]
        .set_index("instance_key")["objective"]
        .to_dict()
    )

    heur = df[df["algo"] != GUROBI_LABEL].dropna(subset=["LB"]).copy()
    if heur.empty:
        return
    heur["true_gap_pct"] = heur.apply(
        lambda r: (100.0 * (g_obj[r["instance_key"]] - r["LB"]) / g_obj[r["instance_key"]])
                  if (r["instance_key"] in g_obj
                      and g_obj[r["instance_key"]] is not None
                      and not pd.isna(g_obj[r["instance_key"]])
                      and g_obj[r["instance_key"]] > 0)
                  else np.nan,
        axis=1,
    )
    heur = heur.dropna(subset=["true_gap_pct"])
    if heur.empty:
        return

    cat_order = [c for c in _INSTANCE_CATEGORY_ORDER if c in heur["category"].unique()]
    heur["category_label"] = pd.Categorical(
        heur["category"].map(_CATEGORY_DISPLAY).fillna(heur["category"]),
        categories=[_CATEGORY_DISPLAY.get(c, c) for c in cat_order],
        ordered=True,
    )

    algos = sorted(heur["algo"].unique())
    palette = {a: _color_for(a) for a in algos}

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(
        data=heur, x="category_label", y="true_gap_pct", hue="algo",
        palette=palette, order=heur["category_label"].cat.categories,
        flierprops=dict(marker=".", markersize=3, alpha=0.4),
        ax=ax,
    )
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Problem category")
    ax.set_ylabel("True optimality gap (%)")
    ax.set_title("Gap Distribution by Category × Heuristic\n"
                 "(boxes show IQR, whiskers 1.5·IQR, dots = outliers)")
    ax.legend(title=None, loc="upper left")
    _save(fig, os.path.join(figures_dir, "fig_gap_boxplots.png"))


def fig_phase_transition_heatmap(df: pd.DataFrame, figures_dir: str) -> None:
    """Goldilocks-zone heatmap: α (rows) × β (cols), value = mean Gurobi runtime
    or failure rate."""
    sub = df[(df["category"] == "goldilocks") & (df["algo"] == GUROBI_LABEL)].copy()
    if sub.empty:
        # Try all categories as a fallback so the figure is still produced.
        sub = df[df["algo"] == GUROBI_LABEL].copy()
        if sub.empty:
            return

    rt = sub.pivot_table(index="alpha", columns="beta",
                         values="runtime_total", aggfunc="mean")
    fail = sub.copy()
    fail["failed"] = ~fail["feasible"]
    fr = fail.pivot_table(index="alpha", columns="beta",
                          values="failed", aggfunc="mean") * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    if not rt.empty:
        sns.heatmap(rt, ax=axes[0], cmap="YlOrRd", annot=True, fmt=".1f",
                    cbar_kws={"label": "Mean runtime (s)"},
                    linewidths=0.4, linecolor="white")
    axes[0].set_title("Gurobi Mean Runtime — α × β")
    axes[0].set_xlabel("Conflict density β")
    axes[0].set_ylabel("Graph density α")

    if not fr.empty:
        sns.heatmap(fr, ax=axes[1], cmap="Blues", annot=True, fmt=".0f",
                    vmin=0, vmax=100,
                    cbar_kws={"label": "Failure / timeout rate (%)"},
                    linewidths=0.4, linecolor="white")
    axes[1].set_title("Gurobi Failure Rate — α × β")
    axes[1].set_xlabel("Conflict density β")
    axes[1].set_ylabel("Graph density α")

    fig.suptitle("Phase Transition (Goldilocks Zone) — Exact Solver Performance",
                 y=1.04)
    _save(fig, os.path.join(figures_dir, "fig_phase_transition_heatmap.png"))


def fig_performance_profile(df: pd.DataFrame, figures_dir: str,
                            tau_max: float = 4.0, n_points: int = 200) -> None:
    """Dolan-Moré performance profile on heuristic LBs.

    Reference per instance = best LB any heuristic achieved on that instance.
    The profile ρ_a(τ) reports the share of instances where a's LB is within
    factor τ of the reference.  Higher / further-left = mathematically better.
    """
    heur = df[df["algo"] != GUROBI_LABEL].dropna(subset=["LB"]).copy()
    if heur.empty:
        return
    pivot = heur.pivot_table(index="instance_key", columns="algo",
                             values="LB", aggfunc="max")
    if pivot.empty or pivot.shape[1] < 2:
        return

    # Performance ratio: r_{p,a} = best_LB_p / LB_{p,a}  (≥ 1, lower is better).
    best_per_instance = pivot.max(axis=1)
    ratios = pivot.apply(lambda col: best_per_instance / col)
    ratios = ratios.replace([np.inf, -np.inf], np.nan)

    # Tau grid in log space.
    taus = np.linspace(1.0, tau_max, n_points)

    fig, ax = plt.subplots(figsize=(10, 6))
    for algo in sorted(pivot.columns):
        col = ratios[algo].dropna().values
        if len(col) == 0:
            continue
        rho = np.array([(col <= t).mean() for t in taus])
        ax.step(taus, rho, where="post", color=_color_for(algo), label=algo,
                linewidth=2.0)

    ax.set_xlabel(r"Performance ratio $\tau$ (factor of best-known LB)")
    ax.set_ylabel(r"Fraction of instances solved within $\tau$")
    ax.set_xlim(1.0, tau_max)
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Dolan–Moré Performance Profile (Heuristic LBs)\n"
                 "higher / further-left dominates")
    ax.legend(loc="lower right")
    _save(fig, os.path.join(figures_dir, "fig_performance_profile.png"))


# =============================================================================
# 6. ORCHESTRATION
# =============================================================================

def _reset_outputs(analysis_dir: str) -> Tuple[str, str]:
    figures_dir = os.path.join(analysis_dir, "figures")
    tables_dir = os.path.join(analysis_dir, "tables")
    if os.path.isdir(figures_dir):
        shutil.rmtree(figures_dir)
    if os.path.isdir(tables_dir):
        shutil.rmtree(tables_dir)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    # Also clear any old top-level workbook so re-runs are clean.
    for f in glob.glob(os.path.join(analysis_dir, "*.xlsx")):
        try:
            os.remove(f)
        except OSError:
            pass
    return figures_dir, tables_dir


def _safe(name: str, fn, *args, **kwargs):
    """Run a table/figure builder; log+continue on error so one failure
    doesn't abort the whole pipeline."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"  ! {name} failed: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Academic-grade analysis pipeline for MAX-APC experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", default="results",
                        help="Directory containing heuristic + Gurobi JSONs.")
    parser.add_argument("--analysis-dir", default="analysis_output",
                        help="Output root for tables, figures, and the Excel workbook.")
    args = parser.parse_args()

    print("==Resetting output directories ==================================")
    figures_dir, tables_dir = _reset_outputs(args.analysis_dir)
    _set_style()

    print("==Loading results ===============================================")
    df = load_master(args.results_dir)
    if df.empty:
        print("No results found. Run batch_experiment.py / gurobi_batch.py first.")
        return

    n_inst = df["instance_key"].nunique()
    n_heur = df[df["algo"] != GUROBI_LABEL]["algo"].nunique()
    has_gurobi = (df["algo"] == GUROBI_LABEL).any()
    print(f"  {len(df):,} rows | {n_inst} unique instances | "
          f"{n_heur} heuristics | Gurobi optima: "
          f"{'yes' if has_gurobi else 'no - true gap will be NaN'}")

    excel_path = os.path.join(args.analysis_dir, "analysis_tables.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        # Master long-format sheet for transparency.
        raw = df.copy()
        for col in ("alpha", "beta", "objective", "runtime_total",
                    "LB", "UB", "first_feasible_time"):
            if col in raw.columns:
                raw[col] = raw[col].round(4)
        _export_table(raw.drop(columns=["instance_key"], errors="ignore"),
                      "raw_long_form", tables_dir,
                      caption="All result rows (raw long-form).",
                      excel_writer=writer)

        print("\n==Phase 2: Tables ===============================================")
        feas = _safe("feasibility_matrix", table_feasibility_matrix, df)
        _export_table(feas, "feasibility_matrix", tables_dir,
                      caption="Success rate (\\%) per category × algorithm.",
                      excel_writer=writer)

        runtime = _safe("runtime_pivot", table_runtime_pivot, df)
        _export_table(runtime, "runtime_pivot", tables_dir,
                      caption="Mean and standard deviation of runtime (s) per (n, $\\alpha$, $\\beta$).",
                      excel_writer=writer)

        bounds = _safe("bounds_gap", table_bounds_gap, df)
        _export_table(bounds, "bounds_and_gap", tables_dir,
                      caption="Mean LB, UB, and optimality gap per (n, $\\alpha$, $\\beta$, algorithm).",
                      excel_writer=writer)

        wins = _safe("win_rates", table_win_rates, df)
        _export_table(wins, "win_rates", tables_dir,
                      caption="Per-algorithm win rates: best LB and fastest time-to-feasible.",
                      excel_writer=writer)

        wilcoxon = _safe("wilcoxon_pairwise", table_wilcoxon_pairwise, df)
        _export_table(wilcoxon, "wilcoxon_pairwise", tables_dir,
                      caption="Pairwise Wilcoxon signed-rank tests on paired true gaps.",
                      excel_writer=writer)

        gurobi_diff = _safe("gurobi_difficulty", table_gurobi_difficulty, df)
        _export_table(gurobi_diff, "gurobi_difficulty", tables_dir,
                      caption="Gurobi per-instance difficulty: nodes explored, solutions found, gap.",
                      excel_writer=writer)

    print(f"  Saved {excel_path}")

    print("\n==Phase 3: Figures ==============================================")
    _safe("convergence_dynamics", fig_convergence_dynamics,
          df, args.results_dir, figures_dir)
    _safe("runtime_scaling", fig_runtime_scaling, df, figures_dir)
    _safe("gap_boxplots", fig_gap_boxplots, df, figures_dir)
    _safe("phase_transition_heatmap", fig_phase_transition_heatmap, df, figures_dir)
    _safe("performance_profile", fig_performance_profile, df, figures_dir)

    print(f"\n==Done.  All outputs in: {args.analysis_dir}/ ==================")


if __name__ == "__main__":
    main()
