"""
analysis.py
============

Load heuristic results and Gurobi optimal solutions, compute metrics,
and generate scalable comparative visuals for multiple heuristics.

Usage:
    python analysis.py
    python analysis.py --results-dir results --analysis-dir analysis_output
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# -----------------------------------------------------------------------------
# File Management
# -----------------------------------------------------------------------------
def clean_output_directory(analysis_dir: str, figures_dir: str) -> None:
    """Purge old Excel and PNG files to ensure a clean overwrite."""
    if os.path.exists(analysis_dir):
        for f in glob.glob(os.path.join(analysis_dir, "*.xlsx")):
            os.remove(f)
    if os.path.exists(figures_dir):
        for f in glob.glob(os.path.join(figures_dir, "*.png")):
            os.remove(f)
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)


# -----------------------------------------------------------------------------
# Loading and Flattening
# -----------------------------------------------------------------------------
def load_all_results(results_dir: str = "results") -> pd.DataFrame:
    """Load heuristic and optimal results into a unified DataFrame."""
    
    # 1. Load optimal results (Gurobi)
    optimal = {}
    optimal_runtimes = {}
    for fpath in glob.glob(os.path.join(results_dir, "optimal_*.json")):
        with open(fpath, "r") as f:
            data = json.load(f)
        base = os.path.basename(fpath).split("_")
        n = int(base[1][1:])
        density = int(base[2][1:]) / 10000.0
        seed = int(base[3][1:].split(".")[0])
        key = (n, density, seed)
        
        if data.get("status") in ["OPTIMAL", "TIME_LIMIT"]:
            optimal[key] = data.get("objective", None)
            optimal_runtimes[key] = data.get("runtime", None)

    # 2. Load heuristic results
    records = []
    for fpath in glob.glob(os.path.join(results_dir, "result_*.json")):
        with open(fpath, "r") as f:
            data = json.load(f)

        n_val = data["n"]
        density = data["num_conflicts"] / (n_val * n_val) if n_val > 0 else 0.0

        base_info = {
            "n": n_val,
            "seed": data["seed"],
            "num_conflicts": data["num_conflicts"],
            "density": density,
            "heuristic": data["heuristic"],
        }
        
        key = (n_val, density, base_info["seed"])
        base_info["opt_objective"] = optimal.get(key, None)
        base_info["opt_runtime"] = optimal_runtimes.get(key, None)

        subg = data.get("subgradient", {})
        base_info.update({
            "subg_LB": subg.get("LB", None),
            "subg_UB": subg.get("UB", None),
            "subg_runtime": subg.get("runtime_seconds", None),
        })

        heur_out = data.get("heuristic_output", {})

        def calc_metrics(obj, ub, opt):
            metrics = {"true_gap_pct": np.nan, "dual_gap_pct": np.nan}
            if obj is not None and obj > 0 and opt is not None and opt > 0:
                metrics["true_gap_pct"] = max(0.0, 100.0 * (opt - obj) / opt)
            if ub is not None and opt is not None and opt > 0:
                metrics["dual_gap_pct"] = max(0.0, 100.0 * (ub - opt) / opt)
            return metrics

        # Expand orderings if present
        if "ordering_variants" in heur_out:
            for ordering, ord_data in heur_out["ordering_variants"].items():
                rec = base_info.copy()
                rec["ordering_label"] = ord_data.get("ordering_label", ordering)
                rec["Algorithm"] = f"{rec['heuristic']} ({rec['ordering_label']})"
                rec["objective"] = ord_data.get("objective", None)
                rec["feasible"] = ord_data.get("feasible", False)
                rec["heuristic_runtime"] = ord_data.get("runtime_seconds", None)
                rec.update(calc_metrics(rec["objective"], rec["subg_UB"], rec["opt_objective"]))
                records.append(rec)
        else:
            rec = base_info.copy()
            rec["ordering_label"] = "Default"
            rec["Algorithm"] = f"{rec['heuristic']} (Default)"
            rec["objective"] = heur_out.get("objective", None)
            rec["feasible"] = heur_out.get("feasible", False)
            rec["heuristic_runtime"] = heur_out.get("runtime_seconds", None)
            rec.update(calc_metrics(rec["objective"], rec["subg_UB"], rec["opt_objective"]))
            records.append(rec)

    df = pd.DataFrame(records)
    
    # Ensure numerics
    numeric_cols = ["n", "num_conflicts", "density", "subg_UB", "subg_LB",
                    "subg_runtime", "objective", "heuristic_runtime", "opt_objective", 
                    "opt_runtime", "true_gap_pct", "dual_gap_pct"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill feasible NaNs with False just in case
    if "feasible" in df.columns:
        df["feasible"] = df["feasible"].fillna(False).astype(bool)

    return df


# -----------------------------------------------------------------------------
# Table Generators
# -----------------------------------------------------------------------------
def get_table_overall_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Table 1: Overall aggregate performance per heuristic algorithm."""
    agg_dict = {
        "n_instances": ("objective", "count"),
        "feasibility_%": ("feasible", lambda x: x.mean() * 100),
        "mean_true_gap_%": ("true_gap_pct", "mean"),
        "mean_dual_gap_%": ("dual_gap_pct", "mean"),
        "heur_time_s": ("heuristic_runtime", "mean")
    }
    summary = df.groupby(["Algorithm"]).agg(**agg_dict).reset_index()
    return summary.round(2)


def get_table_win_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Table 2: Win rate (%) - how often each heuristic found the strictly best solution."""
    df_feas = df[df["feasible"] == True].copy()
    if df_feas.empty:
        return pd.DataFrame()
    
    max_objs = df_feas.groupby(["n", "density", "seed"])["objective"].transform("max")
    df_feas["is_winner"] = (df_feas["objective"] == max_objs)
    
    win_rates = df_feas.groupby("Algorithm")["is_winner"].mean() * 100
    return win_rates.reset_index(name="Win_Rate_%").round(1).sort_values(by="Win_Rate_%", ascending=False)


# -----------------------------------------------------------------------------
# Plotting Functions
# -----------------------------------------------------------------------------
def set_plot_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams["figure.figsize"] = (9, 6)
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.family"] = "sans-serif"


def plot_gap_trends(df: pd.DataFrame, figures_dir: str):
    """Figure 1: Line plots of True Gap vs N, grouped by Heuristic."""
    set_plot_style()
    df_valid = df[df["true_gap_pct"].notna()]
    if df_valid.empty: return
    
    plt.figure()
    sns.lineplot(data=df_valid, x="n", y="true_gap_pct", hue="Algorithm", marker="o", errorbar=None)
    plt.title("Closeness to Optimal: True Gap vs. Problem Size")
    plt.xlabel("Problem Size (n)")
    plt.ylabel("True Gap (%)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig1_true_gap_trend.png"))
    plt.close()


def plot_feasibility_heatmap(df: pd.DataFrame, figures_dir: str):
    """Figure 2: Heatmap showing variation of solvability depending on size and density."""
    set_plot_style()
    if df.empty: return

    # Calculate feasibility percentage per (Algorithm, n, density)
    feas_df = df.groupby(["Algorithm", "n", "density"])["feasible"].mean().reset_index()
    feas_df["feasible_pct"] = feas_df["feasible"] * 100

    # Facet grid for multiple algorithms
    g = sns.FacetGrid(feas_df, col="Algorithm", col_wrap=3, height=4)
    
    def draw_heatmap(data, **kwargs):
        pivot = data.pivot(index="n", columns="density", values="feasible_pct")
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", cbar=False, vmin=0, vmax=100, **kwargs)
        
    g.map_dataframe(draw_heatmap)
    g.set_axis_labels("Conflict Density", "Problem Size (n)")
    g.fig.suptitle("Solvability (Feasibility %) Variation by Size & Density", y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig2_feasibility_heatmap.png"))
    plt.close()


def plot_win_rates(df: pd.DataFrame, figures_dir: str):
    """Figure 3: Bar chart of Algorithm Win Rates."""
    set_plot_style()
    win_df = get_table_win_rates(df)
    if win_df.empty: return

    plt.figure(figsize=(10, max(5, len(win_df) * 0.5)))
    sns.barplot(data=win_df, x="Win_Rate_%", y="Algorithm", hue="Algorithm", palette="viridis", legend=False)
    plt.title("Heuristic Win Rates (Frequency of Best Objective Found)")
    plt.xlabel("Win Rate (%)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig3_win_rates.png"))
    plt.close()


def plot_runtime_scaling(df: pd.DataFrame, figures_dir: str):
    """Figure 4: Log-scale runtime comparison."""
    set_plot_style()
    
    # Isolate exact and dual bounding times (unique per instance)
    base_df = df.drop_duplicates(subset=["n", "density", "seed"]).copy()
    rt_data = base_df.groupby("n")[["opt_runtime", "subg_runtime"]].mean().reset_index()
    
    # Add heuristic runtimes
    heur_rt = df.groupby(["n", "Algorithm"])["heuristic_runtime"].mean().reset_index()
    
    plt.figure()
    
    # Plot baseline bounds
    sns.lineplot(data=rt_data, x="n", y="opt_runtime", label="Gurobi (Optimal)", color="black", linestyle="--", marker="s")
    sns.lineplot(data=rt_data, x="n", y="subg_runtime", label="Subgradient (UB)", color="gray", linestyle=":", marker="d")
    
    # Plot each heuristic runtime
    sns.lineplot(data=heur_rt, x="n", y="heuristic_runtime", hue="Algorithm", marker="o")
    
    plt.yscale("log")
    plt.title("Time Complexity: Runtime Scaling (Log Scale)")
    plt.xlabel("Problem Size (n)")
    plt.ylabel("Average Runtime (Seconds)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig4_runtime_scaling.png"))
    plt.close()


# -----------------------------------------------------------------------------
# Main Driver
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Analyse MAX-APC results.")
    parser.add_argument("--results-dir", default="results", help="Directory with JSON files")
    parser.add_argument("--analysis-dir", default="analysis_output", help="Root output directory")
    args = parser.parse_args()

    figures_dir = os.path.join(args.analysis_dir, "figures")
    
    print("Cleaning old outputs to overrun previous files...")
    clean_output_directory(args.analysis_dir, figures_dir)

    print("Loading results and optimal solutions...")
    df = load_all_results(args.results_dir)
    if df.empty:
        print("No results found. Run batch_experiment.py first.")
        return

    print(f"Loaded data for {df['n'].count()} experimental runs across {df['Algorithm'].nunique()} heuristics.")

    print("\nGenerating tables...")
    tables = {
        "1_Overall_Summary": get_table_overall_summary(df),
        "2_Algorithm_Win_Rates": get_table_win_rates(df),
    }

    excel_path = os.path.join(args.analysis_dir, "analysis_tables.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for name, sheet_df in tables.items():
            if not sheet_df.empty:
                sheet_df.to_excel(writer, sheet_name=name[:31], index=False)
    print(f"  Saved {excel_path}")

    print("\nGenerating scalable figures...")
    plot_gap_trends(df, figures_dir)
    plot_feasibility_heatmap(df, figures_dir)
    plot_win_rates(df, figures_dir)
    plot_runtime_scaling(df, figures_dir)
    
    print(f"\nAnalysis complete. Clean outputs saved in: {args.analysis_dir}/")

if __name__ == "__main__":
    main()