# MAX-APC Solver Framework

This framework is designed to solve the Maximum Assignment Problem with Conflict pair constraints. It uses a modular architecture to compare exact solutions, dual bounds from Lagrangian relaxation, and primal bounds from heuristics.

## System Structure

The project is organized into functional modules that handle different stages of the optimization pipeline:

* **apc_base.py**: The core engine containing the Hungarian algorithm for subproblems and the subgradient optimization loop. It manages the mathematical transition between the original problem and the Lagrangian relaxation.
* **instance_generator.py**: Creates individual test problems. Every instance is generated with a guaranteed feasible solution called E0, which has a weight of zero. This ensures that heuristics always have a baseline for comparison.
* **generate_many_instances.py**: A script used for large-scale data generation. It automates the creation of 1000 instances by randomly selecting parameters from a predefined range of problem sizes and conflict densities.
* **gurobi_solver.py / gurobi_batch.py**: These scripts interface with the Gurobi optimizer to find the mathematically optimal solution for each instance. This provides the ground truth for optimality gap calculations.
* **batch_experiment.py**: The execution controller. It automatically detects all available heuristics, runs them against the instances, and caches subgradient results to avoid redundant calculations.
* **analysis.py**: The reporting tool. It aggregates JSON results from various solvers to calculate true optimality gaps, win rates for different algorithms, and runtime scalability.

## Adding New Heuristics

The framework uses dynamic discovery, which eliminates the need to modify main execution scripts when adding a new algorithm. To implement a new heuristic:

1.  **Create a File**: Add a new Python file inside the `heuristics/` directory (e.g., `my_new_algorithm.py`).
2.  **Define the Interface**: Your file must contain a `run` function or a `run_all_orderings` function with the following signature:
    `def run(x_star, cost_matrix, conflicts, n, E0, **kwargs):`
    * `x_star`: The solution from the Lagrangian subproblem.
    * `cost_matrix`: The original profit values.
    * `conflicts`: The list of forbidden edge pairs.
    * `n`: The problem size.
    * `E0`: The guaranteed feasible starting solution.
3.  **Return Format**: The function must return a tuple containing `(assignment, objective_value, feasibility_status)`.
4.  **Automatic Detection**: Once the file is saved in the `heuristics/` folder, `batch_experiment.py` will identify it and include it in the next experimental run.
5.  **Analysis**: The `analysis.py` script will automatically create new data points for your heuristic in the generated Excel tables and graphs.