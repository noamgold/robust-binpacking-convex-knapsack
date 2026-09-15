# Robust Extensible Bin Packing and Convex Knapsack

Research code accompanying *Robust Extensible Bin Packing and a Convex
Knapsack Problem* by Noam Goldberg, Michael Poss, and Yariv N. Marmor.

The repository contains the algorithms used for the paper's computational
study: a pseudo-polynomial dynamic program (DP), an SOS2/Gurobi formulation
for the two-piece convex knapsack separation problem, and a Pyomo/Gurobi
row-and-column generation solver for robust extensible bin packing (REBP).

## Repository Layout

```text
src/
  knapsack.py       Binary-knapsack DP primitives and synthetic instances
  sos2.py           SOS2 model and convex-knapsack DP algorithms
  rebpp.py          REBP master, separation, and scenario generation
  rbptest_grb.py    Runtime benchmark on Song et al. instances
data/
  30/, 60/, 90/    Benchmark instances used by rbptest_grb.py
  ...               Remaining paper data and auxiliary analysis files
```

The active implementation is kept once, under `src/`. This avoids divergent
copies of the same algorithm. Generated schedules, caches, and benchmark
instance files created by experiments are not source inputs.

## Mathematical Mapping

The code implements the paper's uncertainty set

$$
U_\Omega = \{a : 0 \le a \le \hat a,\ \sum_i a_i \le \Omega\}.
$$

For a bin assignment, the REBP model uses nominal durations `a_bar`, maximum
deviations `a_hat`, shift capacity `V`, overtime costs `c`, assignment
variables `z`, opening variables `y`, and worst-case overtime variable
`theta`. These correspond directly to formulation (1) in the paper.

`src/rebpp.py` alternates between:

1. Solving the finite-scenario Pyomo master problem.
2. Solving the separation problem `eta*` for the current assignment.
3. Adding a violating scenario and repeating until `eta* <= theta + tol`.

For an integral assignment, the separation problem is the two-piece convex
knapsack (2PCK). In `src/sos2.py`, each row of `p` and `b` represents the
breakpoint profits and weights of one convex piecewise-linear function.
`convex_pw_knapsack_dp_profit` implements Algorithm 2 using the binary-knapsack
recurrence

$$
\zeta_f(P,k) = \text{minimum weight of a subset of } [k]\setminus\{f\}
\text{ with profit at least } P.
$$

`convex_pw_knapsack_dp` implements the weight-indexed variant from Appendix A,
where `Pi_f(U,k)` stores the maximum profit attainable with weight at most
`U`. `sos2_gurobi` implements formulation (16) with Pyomo SOS2 variables.

## Environment

The project uses Python 3.12 and a local virtual environment at `.venv`.
The optimization modules require:

- NumPy
- Pandas
- Numba
- Pyomo
- gurobipy
- PySCIPOpt
- packaging

Gurobi is required for the SOS2 and REBP experiments. A valid Gurobi license
must be available in the environment.

## Running the Code

From the repository root:

```bash
.venv/bin/python -m py_compile src/knapsack.py src/sos2.py src/rebpp.py src/rbptest_grb.py
```

Run the case-study solver:

```bash
.venv/bin/python src/rebpp.py
```

Run the long benchmark experiment:

```bash
.venv/bin/python src/rbptest_grb.py
```

Run the convex-knapsack comparison experiment:

```bash
.venv/bin/python src/sos2.py
```

`sos2.py` compares SOS2/Gurobi values against the DP reference and raises
`Exception: different obj vals` if a discrepancy is detected. This is an
algorithm-validation failure, not a dependency-installation error.

## Data Availability

The benchmark data under `data/30`, `data/60`, and `data/90` are used by
`rbptest_grb.py` and are attributed in `data/readme`. The original
`Dep13300with_a_ahat...` case-study files were intentionally removed from the
`TEST` branch in commit `e9eeb27`; consequently `src/rebpp.py` retains the
paper-era path but cannot execute that case study until those inputs are
restored.

## Reproducibility Notes

- `sos2.py` seeds Python's random generator with `101` in its experiment block.
- Numba may spend additional time compiling kernels on the first invocation.
- Gurobi timing is machine- and license-environment-dependent.
- The benchmark script is intentionally long-running and uses the time limits
  defined in the source.

## Citation

Goldberg, N., Poss, M., and Marmor, Y. N. *Robust Extensible Bin Packing and
a Convex Knapsack Problem*, August 10, 2026.