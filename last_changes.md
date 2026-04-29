Release: MAX-APC Memory Optimizations and Evaluation Refactor
This update drastically reduces the memory footprint of the MAX-APC heuristic suite, introduces dynamic fair-benchmarking, and expands the instance generation pipeline to better stress-test the exact solver.

Memory and Performance Optimizations (Approx. 10x RAM Reduction)
Dense Adjacency Lists: Replaced heavy Python Dict[(i,j), Set[(i,j)]] conflict structures with 1D numpy int32 arrays (build_conflict_adjacency_int).

Boolean Bitmasks: All heuristics (lagrangean_repair, lagrangean_repair_2, lagrangean_lambda) now use O(1) boolean bitmasks for set tracking instead of Python Sets.

In-Place Subgradient Math: Downcasted arrays to float32/int32, pre-allocated buffers, and used in-place operators (+=, np.maximum(out=...)) to prevent temporary array allocation churn.

Garbage Collection: Added explicit gc.collect() at the end of every heuristic run to prevent dead-object accumulation on large batches, plus an optional --tracemalloc flag for memory profiling.

Dynamic Execution and Fair Benchmarking
Removed Caching: Eliminated the shared subgradient_cache. Each heuristic now independently computes its own Lagrangian dual bounds in memory, ensuring rigorous, fair CPU wall-clock comparisons.

Time Limits: Enforced a strict 600-second wall-clock time limit for both the Gurobi exact solver and the heuristic subgradient loop.

Convergence Tracking: The subgradient solver now records a full iteration history (LB, UB, step-size, elapsed time), saved natively in the JSON payload for plotting convergence graphs and duality gaps.

Instance Generation and Stress Tests
The Alpha Parameter: Promoted graph density (alpha) to a primary experimental axis alongside n and beta.

Standard Grid: Consolidated standard generation into a unified 640-instance factorial grid.

Stress-Test Categories: Restructured difficult instances into three distinct profiles to specifically target solver weaknesses:

Goldilocks Zone: Sweeps the phase-transition boundary to force Branch and Bound tree explosions.

Degeneracy: Uses flat costs (95-100) to defeat Gurobi's LP-relaxation pruning.

Extreme Scale: Pushes dimensionality to n=150 while capping alpha <= 0.6 to safely test memory limits.

File Naming and Pipeline: Unified file naming formats (e.g., optimal_n20_a04_b010_s1.json). Both generation and batch-solving scripts now instantly skip already-processed files for seamless start/stop execution.