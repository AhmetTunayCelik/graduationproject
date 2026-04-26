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
# GLOBAL GRAPH PARAMETERS
# ---------------------------------------------------------
ALPHA = 1.0  # 1.0 = Complete graph (dense). Set < 1.0 for sparse.
DEFAULT_COST_LOW = 1
DEFAULT_COST_HIGH = 100

# ---------------------------------------------------------
# GENERATOR 1: STANDARD INSTANCES (generate_many_instances.py)
# ---------------------------------------------------------
# Small instances (Wide beta range, high replication)
STD_SMALL_N = [10, 20]
STD_SMALL_BETAS = [0.001, 0.005, 0.010, 0.020, 0.050]
STD_SMALL_SEEDS = list(range(1, 11))  # 10 seeds

# Medium instances
STD_MED_N = [30, 50]
STD_MED_BETAS = [0.0002, 0.0005, 0.001, 0.003, 0.008]
STD_MED_SEEDS = list(range(1, 8))     # 7 seeds

# Large instances (Tight beta range, lower replication)
STD_LARGE_N = [75, 100]
STD_LARGE_BETAS = [0.00005, 0.0001, 0.00015, 0.0002, 0.0003]
STD_LARGE_SEEDS = list(range(1, 6))   # 5 seeds

# ---------------------------------------------------------
# GENERATOR 2: DIFFICULT INSTANCES (generate_difficult_instances.py)
# ---------------------------------------------------------
# Phase Transition (High Beta to explode Branch & Bound)
DIFF_GOLDILOCKS_N = [40, 50]
DIFF_GOLDILOCKS_BETAS = [0.01, 0.03, 0.05, 0.08, 0.12, 0.15]
DIFF_GOLDILOCKS_SEEDS = list(range(1, 6))

# Degeneracy (Flat costs to break LP pruning)
DIFF_DEGEN_N = [30, 40, 50]
DIFF_DEGEN_BETAS = [0.005, 0.01, 0.02, 0.05]
DIFF_DEGEN_SEEDS = list(range(1, 6))
DIFF_DEGEN_COST_LOW = 95
DIFF_DEGEN_COST_HIGH = 100

# Extreme Scale (Testing pure dimensionality limits)
DIFF_EXTREME_N = [100, 150, 200]
DIFF_EXTREME_BETAS = [0.00005, 0.0001, 0.0002]
DIFF_EXTREME_SEEDS = list(range(1, 4))

# ---------------------------------------------------------
# EXACT SOLVER (GUROBI)
# ---------------------------------------------------------
GUROBI_TIME_LIMIT = 300.0  # 5 minutes maximum runtime
GUROBI_THREADS = 0         # 0 = use all available cores
GUROBI_OUTPUT_FLAG = 0     # 0 = quiet, 1 = verbose

# ---------------------------------------------------------
# LAGRANGEAN SUBGRADIENT SETTINGS
# ---------------------------------------------------------
SUBG_MAX_ITERS = 500
SUBG_STAGNATION_LIMIT = 20 # Halve step size after 20 non-improving iters
SUBG_EPSILON = 1e-6        # Tolerance for subgradient norm