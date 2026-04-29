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
UNIVERSAL_SEEDS = list(range(1, 6))    # 5 seeds is the academic minimum for statistical validity
GRAPH_DENSITIES = [0.4, 1.0]           # Test only sparse (40%) and dense (100%) graphs
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
DIFF_GOLDILOCKS_ALPHAS = [0.4, 1.0]    # Reduced from 4
DIFF_GOLDILOCKS_BETAS = [0.15, 0.20, 0.25, 0.30]
# New Size: 2 * 2 * 4 * 5 = 80 instances

# B. Degeneracy (The "Flat Cost" Trap)
# Tight cost band defeats Gurobi's LP-bound pruning.
DIFF_DEGEN_N = [30, 40, 50]
DIFF_DEGEN_ALPHAS = [0.4, 1.0]         # Reduced from 4
DIFF_DEGEN_BETAS = [0.05, 0.10, 0.15]
DIFF_DEGEN_COST_LOW = 95
DIFF_DEGEN_COST_HIGH = 100
# New Size: 3 * 2 * 3 * 5 = 90 instances

# C. Extreme Scale (Pure Dimensionality Limits)
# Alpha capped at 0.6 to keep conflict-pair pool tractable at n=150.
DIFF_EXTREME_N = [100, 150]
DIFF_EXTREME_ALPHAS = [0.2, 0.6]       # Reduced from 3
DIFF_EXTREME_BETAS = [0.001, 0.005]
DIFF_EXTREME_SEEDS = list(range(1, 6)) 
# New Size: 2 * 2 * 2 * 5 = 40 instances


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