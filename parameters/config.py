"""
parameters/config.py
====================
Centralized parameters for the MAX-APC optimization framework.
This file holds the settings for standard data generation, stress-test 
generation, the subgradient loop, and the exact solver.
"""

# ---------------------------------------------------------
# DIRECTORIES
# ---------------------------------------------------------
INSTANCE_DIR = "instances"
RESULTS_DIR = "results"
ANALYSIS_DIR = "analysis_output"

# ---------------------------------------------------------
# GLOBAL PARAMETERS
# ---------------------------------------------------------
UNIVERSAL_SEEDS = list(range(1, 11))   # Exactly 10 seeds for statistical balance
GRAPH_DENSITIES = [0.4, 0.6, 0.8, 1.0] # The alpha parameter (graph edge density)
DEFAULT_COST_LOW = 1
DEFAULT_COST_HIGH = 100

# ---------------------------------------------------------
# GENERATOR 1: STANDARD INSTANCES (Unified Grid)
# ---------------------------------------------------------
GRID_N = [20, 30, 40, 50]
GRID_BETAS = [0.01, 0.05, 0.10, 0.15]

# ---------------------------------------------------------
# GENERATOR 2: DIFFICULT INSTANCES
# ---------------------------------------------------------

# A. The Goldilocks Zone (Branch & Bound Explosion)
# Phase-transition sweep: highly constrained but feasible region.
DIFF_GOLDILOCKS_N = [40, 50]
DIFF_GOLDILOCKS_ALPHAS = [0.4, 0.6, 0.8, 1.0]
DIFF_GOLDILOCKS_BETAS = [0.15, 0.20, 0.25, 0.30]
# Uses UNIVERSAL_SEEDS (10 seeds)

# B. Degeneracy (The "Flat Cost" Trap)
# Tight cost band defeats Gurobi's LP-bound pruning.
DIFF_DEGEN_N = [30, 40, 50]
DIFF_DEGEN_ALPHAS = [0.4, 0.6, 0.8, 1.0]
DIFF_DEGEN_BETAS = [0.05, 0.10, 0.15]
DIFF_DEGEN_COST_LOW = 95
DIFF_DEGEN_COST_HIGH = 100
# Uses UNIVERSAL_SEEDS (10 seeds)

# C. Extreme Scale (Pure Dimensionality Limits)
# Alpha capped at 0.6 to keep conflict-pair pool tractable at n=150.
DIFF_EXTREME_N = [100, 150]
DIFF_EXTREME_ALPHAS = [0.2, 0.4, 0.6]
DIFF_EXTREME_BETAS = [0.001, 0.005]
DIFF_EXTREME_SEEDS = list(range(1, 6))  # 5 seeds to prevent memory crashes

# ---------------------------------------------------------
# EXACT SOLVER (GUROBI)
# ---------------------------------------------------------
GUROBI_TIME_LIMIT = 600.0     # 10 minutes maximum runtime
HEURISTIC_TIME_LIMIT = 600.0  # 10 minutes maximum runtime (subgradient + repair)
GUROBI_THREADS = 0         # 0 = use all available cores
GUROBI_OUTPUT_FLAG = 0     # 0 = quiet, 1 = verbose

# ---------------------------------------------------------
# LAGRANGEAN SUBGRADIENT SETTINGS
# ---------------------------------------------------------
SUBG_MAX_ITERS = 500
SUBG_STAGNATION_LIMIT = 20 # Halve step size after 20 non-improving iters
SUBG_EPSILON = 1e-6        # Tolerance for subgradient norm