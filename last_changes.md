Bug Fixes and Correctness Improvements (post-last-pull) p

Fix #1 — Instance filename existence check (instance_generator.py, generate_difficult_instances.py)
The existence check now uses the realized (back-computed) beta density for the filename, not the requested one. Previously, integer rounding of num_conflicts could produce a different beta tag than expected, causing the check to report "not found" and overwrite valid files on re-runs.

Fix #2 — Convergence figure time alignment (analysis.py)
fig_convergence_dynamics now interpolates each seed's (LB, UB) trace onto a shared 200-point time grid using np.interp, instead of aligning by iteration index K. Averaging by iteration index produced misleading aggregates when runs of the same (n, beta) finished at different wall-clock times.

Fix #4 — SAVLR cost_range inflation on degen instances (heuristics/lagrengean_repair_f.py)
The cost_range filter in _compute_dual_weight and _repair_multi_rho was changed from cost > -1e10 to cost > 0. The old filter included E0 edges (cost=0), inflating cost_range from ~5 to ~101 on degen instances (costs in [95,100]), causing over-aggressive rho and an overly loose dual weight cap.

Fix #6 — Docstring note on hybrid stopping criterion (apc_base.py)
Added a note to subgradient_solve that K_max fires for small n (subsecond) while time_limit fires for large n. Cross-n comparisons of final bound quality should group runs by terminated_reason.

Fix #7 — Subgradient pi_k step-size reset (apc_base.py)
pi_k no longer halves indefinitely toward zero. When pi_k would drop below 1e-6, it resets to 2.0 / 2^k (k = reset count, max 3 resets: 1.0 → 0.5 → 0.25). This prevents the step size from collapsing and stalling convergence on long runs.

Fix #8 — SAVLR cache key id recycling risk (heuristics/lagrengean_repair_f.py)
The module-level cache key now includes the first and last conflict tuples in addition to id(conflicts) and len. This prevents a stale cache hit if Python recycles the memory address of a freed conflicts list from a previous instance.

Fix #9 — SAVLR feasibility missing row/col uniqueness check (heuristics/lagrengean_repair_f.py)
Both _repair_multi_rho and repair() now check that the n completed edge IDs span n distinct rows and n distinct columns, not just that there are n unique edge IDs. Two distinct edge IDs can share a row (e.g. (0,0) and (0,1)), which would pass the old check but is not a valid assignment.

Fix #10 — Gurobi result schema completeness (gurobi_solver.py)
The INFEASIBLE and UNKNOWN return branches now include nodes_explored and solutions_found fields, matching the schema returned by the OPTIMAL and TIME_LIMIT branches and preventing KeyError in downstream analysis.

Fix #11 — find_violations avoids repeated allocation (apc_base.py)
find_violations now accepts optional pre-computed c_e1_flat and c_e2_flat arrays. Callers that already hold these arrays (e.g. the subgradient loop) can pass them in to skip the np.array(conflicts) allocation on every call.

Committed (e90108b) — repair_f integration (analysis.py, heuristics/lagrengean_repair_f.py)
Added lagrangean_repair_savlr to _ALGO_COLORS in analysis.py. Added graph_edge_mask constraint to the feasibility check in repair() so sparse-graph instances (alpha < 1) reject assignments that use non-graph edges.

---

 Major Features & New HeuristicsNew Heuristic: Iterative Tabu (lagrangean_iterative_tabu.py)Implemented a non-greedy, global repair strategy ("Iterative Hungarian with Dynamic Tabu Penalties").Dynamically applies a massive penalty ($M=10000$) to conflicting edges and re-solves the $N \times N$ matrix using scipy.optimize.linear_sum_assignment.Includes a strict 5 micro-iteration limit to prevent infinite Tabu cycling in dense phase-transitions.Maintains the $O(n^4)$ memory optimization utilizing 1D integer Edge IDs and boolean bitmasks.New Control Baseline (greedy_baseline.py)Added a pure greedy heuristic lacking Lagrangian dual information to serve as the scientific control group.Implemented the SKIP_SUBGRADIENT = True flag in batch_experiment.py to bypass the 10-minute subgradient loop entirely for this baseline.📊 Automated Analysis Pipeline (analysis.py)Complete Rewrite for Publication-Grade Metrics:Data Ingestion: Implemented strict NaN handling. feasible_found=False or incumbent_objective=null or Gurobi SolCount==0 now correctly map to np.nan to preserve mathematical integrity in conditional averages.Automated LaTeX Tables: Added auto-generation for Feasibility Matrix, Runtime Pivots, Bounds/Gap comparisons, and Win-rates (via df.to_latex()).Advanced Statistical Testing: Integrated Wilcoxon signed-rank tests for pairwise heuristic comparisons.Advanced Visualizations: Added Dolan-Moré Performance Profiles, phase-transition heatmaps, log-scaled runtime scaling, and subgradient convergence dynamics.Stable Visuals: Created _color_for() to ensure consistent colorblind-friendly palettes for algorithms across all generated figures.🛠️ Critical Architectural FixesSubgradient Lambda Pass-Through: Fixed a critical data flow bug where lambdas were not being passed to repair_fn during the subgradient loop, unlocking the actual dual-aware capabilities of lagrangean_repair_lambda.Post-Loop Incumbent Updates: save_result() now checks if the post-loop heuristic call produced a superior assignment using converged $\lambda$ values, correctly updating the headline incumbent_objective.Single-Threaded Fairness: Set model.setParam("Threads", 1) in gurobi_solver.py to enforce strict single-core execution, ensuring an academically fair wall-clock CPU comparison against the heuristics.Atomic Writes: Replaced standard JSON dumping with atomic file writing (.tmp followed by os.replace()) in both batch runners to prevent file corruption and permanent data loss during mid-write process kills.Data Validation: Added an explicit assert LB <= UB invariant check at the termination of the subgradient loop to catch silent bound-crossing bugs.⚙️ Configuration & Data ManagementLocal Execution Grid Scaling: Reduced the total experimental grid from 1,380 to 370 instances across 4 categories (Standard, Goldilocks, Degeneracy, Extreme) to enable feasible local batch execution.Reproducibility Metadata: Added write_metadata() to capture OS, CPU, RAM, Python/NumPy versions, and timestamps (metadata.json) per batch run.JSON Null Serialization: Refactored apc_base.py JSON exports to explicitly use null instead of 0.0 for failed runs, completely removing the $E_0$ zero-cost data contamination risk.Filtering & Documentation: Fixed the legacy instance filter key to conflict_graph_density and documented the boundary asymmetry of max_valid_pairs for graphs larger than $n=40$.

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