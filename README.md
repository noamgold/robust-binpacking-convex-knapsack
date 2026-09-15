# Robust Extensible Bin Packing and Revisiting the Convex Knapsack Problem

## Abstract

This repository implements algorithms for a robust extensible bin-packing problem. Items with uncertain sizes are assigned to bins with nominal capacity, while capacity violations are permitted through an extensible convex penalty. The outer optimization chooses bin assignments and activated bins; an adversarial inner problem selects item-size deviations from a budgeted ambiguity set. For integral assignments, this separation problem is equivalent to a two-piece convex knapsack problem.

The implementation combines row-and-column generation for the robust bin-packing master problem with dynamic-programming and SOS2/Gurobi solution methods for the convex-knapsack separation problem. The code supports computational comparisons of exact DP methods, SOS2 formulations, and robust bin-packing scenario generation.

## Mathematical Formulation

Let $[m]$ denote the items and $[n]$ the candidate bins. Item $i$ has nominal size $\bar{a}_i$ and maximum deviation $\hat{a}_i$. The budgeted ambiguity set is

$$
\mathcal{U}_\Omega = \{a \in \mathbb{R}^{m} : 0 \le a_i \le \hat{a}_i,\ \sum_{i\in[m]} a_i \le \Omega\}.
$$

Let $Z^{*}=(z^{*}_{ij})$ be the item-to-bin assignment matrix and $Y^{*}=(y^{*}_j)$ the bin-activation vector. For bin $j$, define

$$
B_j(Z^{*}) = \{i\in[m] : z^{*}_{ij}=1\}.
$$

With nominal bin capacity $V$ and overtime coefficient $c_j$, the extensible bin cost under scenario $a$ is

$$
f(B_j,a) = \mathbf{1}_{\{B_j\neq\varnothing\}} + c_j(\sum_{i\in B_j}(\bar{a}_i+a_i)-V)_{+},
$$

where $(x)_{+}=\max\{x,0\}$. The robust min-max problem is

$$
\min_{Y,Z}\ \max_{a\in\mathcal{U}_\Omega}
\sum_{j\in[n]} f\bigl(B_j(Z),a\bigr).
$$

The master problem minimizes the number of activated bins plus an auxiliary worst-case cost variable $\Theta^{*}$:

$$
\min_{Y,Z,\Theta}\ \sum_{j\in[n]} y_j + \Theta.
$$

During row-and-column generation, the current master solution $(Y^{*},Z^{*},\Theta^{*})$ is passed to the separation problem

$$
\eta^{*}(Y^{*},Z^{*}) = \max_{a\in\mathcal{U}_\Omega}
\sum_{j\in[n]} c_j
(\sum_{i\in[m]}z^{*}_{ij}(\bar{a}_i+a_i)-Vy^{*}_j)_{+}.
$$

If $\eta^{*}>\Theta^{*}$, the adversarial scenario is added to the finite master scenario set. Otherwise, the current master solution satisfies the separation test up to numerical tolerance.

For an integral assignment, the separation problem is a two-piece convex knapsack:

$$
\max_{x}\{\sum_{j\in[n]} p_j(x_j):
\sum_{j\in[n]}x_j\le\Omega,\ 0\le x_j\le u_j\}.
$$

where

$$
p_j(x_j)=(\gamma_j+\beta_jx_j)_{+},
$$

$$
\gamma_j=c_j(\sum_{i\in B_j}\bar{a}_i-V),\qquad
\beta_j=c_j,\qquad
u_j=\sum_{i\in B_j}\hat{a}_i.
$$

## Repository Structure

```text
src/
  rebpp.py          REBP master problem, separation, and scenario generation
  knapsack.py       Binary-knapsack dynamic-programming primitives
  sos2.py           SOS2/Gurobi and convex-knapsack DP methods
  rbptest_grb.py    REBP benchmark execution script

data/
  30/               Benchmark instances with 30-item scale
  60/               Benchmark instances with 60-item scale
  90/               Benchmark instances with 90-item scale
  ...               Additional benchmark and auxiliary data files
```

The `src/` modules form the active implementation. The `data/` directory contains the benchmark instance families used by the runtime experiments and auxiliary input data used by the research workflow.

## Dependencies

- Python 3.x, tested with Python 3.12
- NumPy
- Pandas
- Numba
- Pyomo
- Gurobi Optimizer and `gurobipy`
- PySCIPOpt for legacy compatibility paths
- `packaging`

The SOS2 and robust master formulations require a valid Gurobi installation and license. Runtime comparisons are hardware- and solver-configuration-dependent; record the Gurobi version, thread configuration, CPU, and license environment when reproducing computational results.

## Quick Start

From the repository root, use the project virtual environment if available:

```bash
cd /home/rokachda/binpacking_summer_project
.venv/bin/python src/rbptest_grb.py
```

With an activated Python environment, the equivalent command is:

```bash
python src/rbptest_grb.py
```

`rbptest_grb.py` loads benchmark instances through relative paths under `data/`, constructs the REBP parameters, and calls the Gurobi-based master/separation workflow. The benchmark is a long-running experiment rather than a short smoke test.

For syntax validation:

```bash
.venv/bin/python -W error -m py_compile \
  src/knapsack.py src/sos2.py src/rebpp.py src/rbptest_grb.py
```
