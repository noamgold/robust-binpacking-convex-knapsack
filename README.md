# Robust Extensible Bin Packing and Revisiting the Convex Knapsack Problem

Research-code documentation for the paper:

> Noam Goldberg, Michael Poss, and Yariv N. Marmor, *Robust Extensible Bin Packing and a Convex Knapsack Problem*, August 10, 2026.



## Abstract / Overview

This repository implements the computational framework from the paper. The
outer problem assigns uncertain-duration items to extensible bins and minimizes
the number of opened bins plus a worst-case overtime penalty. The inner
adversary selects a scenario from the budgeted ambiguity set
$\mathcal{U}_\Omega$, producing the largest aggregate convex overtime cost.

The resulting min-max problem is solved by row-and-column generation. A finite
scenario master problem proposes an assignment, while a separation oracle
searches for a worst-case scenario. For integral assignments, the oracle is a
two-piece convex knapsack problem. The repository provides both a dynamic
programming oracle and an SOS2/Gurobi formulation, allowing exact-value and
runtime comparisons at the algorithmic level.

The extensible capacity is central: a bin has nominal capacity $V$, but it may
exceed $V$ at a cost determined by a convex positive-part function. Thus the
model captures the operational trade-off between opening additional shifts and
absorbing uncertainty as overtime.

## 1. Research Scope

The paper studies **robust extensible bin packing (REBP)**. An item represents an appointment or surgery with nominal duration $\bar a_i$ and maximum duration deviation $\hat a_i$. Items are assigned to work shifts, represented as bins. Each bin has a nominal capacity $V$, but overtime is permitted and charged linearly.

The uncertainty set is the continuous budgeted set

$$
U_\Omega = \{a \in \mathbb{R}^{m} : 0 \le a_i \le \hat a_i,\quad \sum_{i=1}^m a_i \le \Omega\}.
$$

The uncertainty budget $\Omega$ limits the total deviation that an adversarial scenario can distribute across all assigned items. This is the key robust-optimization distinction from a nominal bin-packing model.

For a bin $j$, assignment set $B_j$, and scenario $a \in U_\Omega$, the paper defines the cost

$$
f(B_j,a) = \mathbf{1}_{\{B_j \ne \varnothing\}} + c_j(\sum_{i\in B_j}(\bar a_i+a_i)-V)_+,
$$

where $(x)_+ = \max\{x,0\}$. The REBP objective is

$$
\min_{B_1,\ldots,B_n}
\max_{a\in U_\Omega}
\sum_{j=1}^{n} f(B_j,a).
$$

The model therefore trades off the number of opened shifts against worst-case overtime.

## 2. Repository Structure

```text
src/
	knapsack.py       Binary-knapsack DP primitives and synthetic instances
	sos2.py           SOS2 model and convex-knapsack DP algorithms
	rebpp.py          REBP master, separation, and scenario generation
	rbptest_grb.py    REBP runtime benchmark driver
data/
	30/, 60/, 90/    Song et al. benchmark instance families
	README.md         Canonical repository documentation
	...               Auxiliary benchmark and case-study resources
```

The implementation is kept once under `src/`; generated schedules, Numba
caches, and experiment outputs are not source inputs.

## 3. Dependencies and Solvers

The reference environment uses Python 3.12 and `.venv`. The code depends on:

- Python standard library: `math`, `random`, `statistics`, `time`, `typing`,
	and `os`.
- Numerical/data packages: NumPy, Pandas, and Numba.
- Optimization modeling: Pyomo.
- Optimization solvers: Gurobi Optimizer through `gurobipy` and PySCIPOpt for
	legacy compatibility paths.
- `packaging`, required by the Pyomo/Gurobi integration in this environment.

The SOS2 formulation and REBP master require a valid Gurobi installation and
license. Gurobi version, CPU hardware, thread settings, and license type can
materially affect runtime results.

## 4. Mathematical Model and Code Variables

The assignment formulation in the paper uses:

| Paper symbol | Code representation | Meaning |
|---|---|---|
| $\bar a_i$ | `a_bar[i]` | Nominal duration of item $i$ |
| $\hat a_i$ | `a_hat[i]` | Maximum deviation of item $i$ |
| $V$ | `V` | Nominal capacity of each bin/shift |
| $\Omega$ | `Omega` | Total deviation budget |
| $c_j$ | `c[j]` | Overtime cost per unit for bin $j$ |
| $y_j$ | `model.y[j]` | Binary variable indicating that bin $j$ is open |
| $z_{ij}$ | `model.z[i, j]` | Binary assignment of item $i$ to bin $j$ |
| $\theta$ | `model.theta` | Upper bound on total worst-case overtime |
| $\alpha_j(a)$ | `model.alpha[j, scenario]` | Scenario-specific overtime in bin $j$ |
| $\widetilde U_\Omega$ | Scenario constraints in `rebpp.py` | Finite scenario set accumulated by generation |

The Pyomo master in `src/rebpp.py` implements the main structural constraints:

$$
\sum_j z_{ij}=1 \quad \forall i,
$$

$$
z_{ij}\le y_j \quad \forall i,j,
$$

$$
\sum_i z_{ij}(\bar a_i+a_i)\le Vy_j+\alpha_j(a) \quad \forall j,a,
$$

and

$$
\sum_j c_j\alpha_j(a)\le\theta \quad \forall a.
$$

The objective is $\min \sum_j y_j+\theta$. These are formulation (1) in the paper.

## 5. Algorithm 1: Row-and-Column Generation

The model has infinitely many scenario-indexed constraints because $U_\Omega$ is continuous. The implementation follows the paper's row-and-column generation procedure:

1. Start with the nominal scenario $a=0$.
2. Solve the finite-scenario master problem with Pyomo and Gurobi.
3. Given the current assignment $(y^{\ast},z^{\ast},\theta^{\ast})$, solve the separation problem.
4. If the separation value $\eta^{\ast}$ violates $\eta^{\ast}\le\theta^{\ast}$, generate the worst scenario $a^{\ast}$.
5. Add the scenario-dependent variables and constraints to the master.
6. Repeat until no violating scenario remains.

The main control routine is `solve_instance` in `src/rebpp.py`. The master construction is `rebppinit_pyomo`; scenario updates are handled by `update_rebpp_pyomo`; and separation data are assembled by `create_sos_instance` and `convex_pw_knapsack_wrapper`.

The separation objective is Proposition 1 of the paper:

$$
\eta^{\ast} = \max_{a\in U_\Omega}
\sum_j c_j(\sum_i z^{\ast}_{ij}(\bar a_i+a_i)-Vy^{\ast}_j)_+.
$$

A scenario is needed whenever $\eta^{\ast}>\theta^{\ast}$, up to the implementation tolerance `VIOL_TOL`.

### Symmetry breaking

When overtime costs are identical, the paper uses the bin-ordering inequalities

$$
y_1\ge y_2\ge\cdots\ge y_n,
$$

and the additional lower bound

$$
(1-y_{j+1})c(\sum_i\bar a_i+\Omega-jV)\le\theta.
$$

These are controlled by `SYMBREAK` and `VALIDINEQ` in `src/rebpp.py`.

## 6. Separation as Two-Piece Convex Knapsack

For an integral assignment, each bin becomes one convex piecewise-linear function. The paper's Observation 1 defines

$$
\gamma_j = c_j(\sum_{i:z^{\ast}_{ij}=1}\bar a_i-V),
\qquad
\beta_j=c_j,
$$

and

$$
u_j=\sum_{i:z^{\ast}_{ij}=1}\hat a_i.
$$

The separation problem becomes the two-piece convex knapsack (2PCK):

$$
\max_{x}
\{\sum_j p_j(x_j):
\sum_j x_j\le\Omega,\ 0\le x_j\le u_j\},
$$

with

$$
p_j(x_j)=(\gamma_j+\beta_jx_j)_+.
$$

In `src/rebpp.py`, `create_sos_instance` constructs the three breakpoints for each bin: the zero point, the point at which overtime begins, and the full-deviation point. The resulting matrices `p` and `b` are passed to `src/sos2.py`.

## 7. `src/sos2.py`: SOS2 and DP Solvers

This module contains two independent solution paths for 2PCK.

### 7.1 SOS2/Gurobi formulation

The function `sos2_gurobi(p, b, B)` implements the paper's formulation (16). For every item/bin $j$, `t[j,k]` are convex-combination variables over breakpoints $k$:

$$
\sum_k t_{jk}=1,
$$

$$
\sum_{j,k} b_{jk}t_{jk}\le\Omega,
$$

with an SOS2 restriction allowing only adjacent breakpoints to be active. The objective is

$$
\max\sum_{j,k}p_{jk}t_{jk}.
$$

This route uses Pyomo and Gurobi. It is a mathematically direct MIP representation of the convex functions, but its running time may be sensitive to branch-and-bound behavior.

### 7.2 Profit-indexed DP: Algorithm 2

`convex_pw_knapsack_dp_profit` implements the paper's Algorithm 2. It uses the binary-knapsack state

$$
\zeta_f(P,k)=\text{minimum weight of a subset of }[k]\setminus\{f\}
\text{ whose profit is at least }P.
$$

The pivot item $f$ is the possible fractional item identified by the extreme-point structure of 2PCK. The solution is evaluated as

$$
P^{\ast} = \max_{f,P}\{P+\hat p_f(\Omega-\zeta_f(P,n))\},
$$

where $\hat p_f$ is the piecewise value of the pivot item.

The paper proves that sorting by non-increasing second-segment slope allows reuse of DP states and reduces the running time to

$$
O(n(P_{\max}+\log n)),
$$

rather than recomputing a full table for every excluded item.

### 7.3 Weight-indexed DP: Appendix A

`convex_pw_knapsack_dp` implements the $\Omega$-DP variant. Its state is

$$
\Pi_f(U,k)=\text{maximum binary-knapsack profit using weight at most }U,
$$

and its final evaluation follows Appendix A, Eq. (18):

$$
P^{\ast}=\max_{f}\max_{U\in[\Omega]}
\{\Pi_f(U,n)+\hat p_f(\Omega-U)\}.
$$

This version is especially relevant inside REBP because the REBP instance uses fractional overtime costs, making the capacity-indexed formulation natural in the separation routine.

## 8. `src/knapsack.py`: Binary-Knapsack Primitives

This module provides the lower-level recurrences used by both convex-knapsack variants:

- `for_loop_method_all_w`: capacity-indexed DP corresponding to $\Pi_f(U,k)$.
- `for_loop_method_all_p`: profit-indexed DP corresponding to $\zeta_f(P,k)$.
- `*_save_all`: table-building variants used for state reuse.
- `P_upper_bound`: a fractional-knapsack upper bound used to choose a finite $P_{\max}$.
- `generate_random_instance`: the inverse-correlation generator used in the computational experiment.

The module is not a second REBP implementation. It is the exact-DP support layer for `sos2.py`.

## 9. `src/rbptest_grb.py`: REBP Runtime Experiments

`rbptest_grb.py` reproduces the REBP benchmark protocol described in Section 5.2 of the paper. For each input instance it sets

$$
\hat a_i=0.4\bar a_i,
\qquad
\Omega=0.1\sum_i\hat a_i,
$$

$$
V=\frac{1}{8}\sum_i(\bar a_i+\hat a_i),
\qquad
c=\frac{1.5}{V},
$$

and the number of candidate bins to

$$
n=\lceil\frac{2(\sum_i\bar a_i+\Omega)}{V}\rceil.
$$

The files in `data/30`, `data/60`, and `data/90` are the Song et al. benchmark families. The script reports elapsed runtime, master runtime, number of scenario-generation iterations, number of bins, and time-limit counts.

This script is intentionally a long-running experiment. It is not a smoke test.

## 10. Mapping Code to Paper Results

### Table 1: Convex-knapsack timing

The paper compares:

- SOS: `sos2_gurobi`;
- DP (Algorithm 2): `convex_pw_knapsack_dp_profit`;
- $\Omega$-DP: `convex_pw_knapsack_dp`.

The experiment block in `src/sos2.py` generates inverse-correlated instances with $R\in\{10^{2},10^{3},10^{4}\}$ and thirty capacity settings. It records SOS and DP elapsed times and prints aggregate summaries. The paper reports that DP is generally faster and more stable, particularly as the SOS2 formulation becomes difficult for Gurobi.

The validation guard compares the SOS objective with the DP objective. A message such as

```text
Exception: different obj vals
```

means that the two implementations produced different objective values on a generated instance. It is an algorithmic correctness discrepancy requiring investigation; it is not evidence that Gurobi or Python failed to install.

### Tables 2 and 3: REBP scenario-generation timing

These tables correspond to `src/rbptest_grb.py` and `rebpp.solve_instance`. Table 2 uses the easier Song et al. instances with smaller item sizes; Table 3 uses harder instances with $a_{\max}=100$. The reported columns are generated by the timing arrays in `rbptest_grb.py`: average and maximum elapsed time, averages including time-limit runs, master-solve time, iteration counts, and time-limit counts.

The paper reports that symmetry breaking and inequality (5) can substantially reduce runtime, especially on harder instances. In code, the relevant controls are `SYMBREAK`, `VALIDINEQ`, and `ITEMSYMBREAK`; `GAPVAL1` and `GAPVAL2` correspond to the two-stage master optimality gaps $\tau_0$ and $\tau_1$ in Algorithm 1.

### Table 4: Surgery case study

Table 4 compares actual, nominal, and robust schedules for Weeks 3, 7, and 8 under two prediction models, V1 and V2. The quantities are utilization and overtime summaries, evaluated over daily schedules. The model inputs are the case-study vectors $\bar a$, $\hat a$, the shift length $V$, and percentile-derived $\Omega$ values (477, 721, and 971).

The robust schedules in the paper generally improve utilization and reduce worst-case overtime relative to the actual and nominal schedules. The code path for this study is the `__main__` block of `src/rebpp.py`, which reads a compatible case-study CSV, solves `solve_instance`, and writes a schedule CSV.

## 11. Installation and Execution

The repository uses Python 3.12 and the local virtual environment `.venv`. Required packages are NumPy, Pandas, Numba, Pyomo, `gurobipy`, PySCIPOpt, and `packaging`. Gurobi also requires a valid license.

From the repository root:

```bash
.venv/bin/python -m py_compile src/knapsack.py src/sos2.py src/rebpp.py src/rbptest_grb.py
```

Core import check:

```bash
cd src
../.venv/bin/python -c "import knapsack, sos2, rebpp; print('core imports passed')"
```

### Quick Start: REBP benchmark

From the repository root, run the Gurobi benchmark against the instance files
under `data/30`, `data/60`, or `data/90`:

```bash
cd /home/rokachda/binpacking_summer_project
.venv/bin/python src/rbptest_grb.py
```

The script currently uses the benchmark family configured by `num_items` in
`src/rbptest_grb.py` and reports runtime, master runtime, scenario iterations,
bin counts, and time-limit statistics. It is a long-running experiment rather
than a short smoke test.

Other experiment entry points are:

```bash
.venv/bin/python src/sos2.py
.venv/bin/python src/rebpp.py
```

Run the case-study entry point when compatible case-study input files are available:

```bash
.venv/bin/python src/rebpp.py
```

Numba compiles kernels on first use, and Gurobi runtime depends on hardware, solver version, license configuration, and parameter settings. Do not compare timings across machines without recording these conditions.

## 12. Data Provenance

- `data/30`, `data/60`, and `data/90` contain instances retrieved from the KU Leuven RMAP instance collection and used in the Song et al. benchmark protocol.
- The healthcare case-study data were originally associated with the SEE Lab source cited in the paper.

## 13. Citation

Goldberg, N., Poss, M., and Marmor, Y. N. (2026). *Robust Extensible Bin Packing and a Convex Knapsack Problem*. August 10, 2026.

For the benchmark family, also cite:

Song, G., Kowalczyk, D., and Leus, R. (2018). The robust machine availability problem—bin packing under uncertainty. *IISE Transactions*, 50(11), 997–1012.

## 14. API and Maintenance Notes

The public computational API follows a NumPy-documentation style. Every
algorithmic entry point documents its mathematical inputs, outputs, and role
in the paper's algorithm. Type annotations use PEP 484-compatible NumPy and
Python types; Pyomo and Gurobi callback objects are typed as `Any` because
their runtime interfaces are solver-specific.

### Compatibility functions

- `sos2.sos2` is a compatibility wrapper around `sos2_gurobi`.
- `knapsack.knapsack` is retained for historical benchmark callers and routes
	to the active weight-indexed DP.
- `knapsack.for_loop_method_profit` is a legacy profit-state helper; new code
	should use `for_loop_method_all_p`.

### Experimental controls

The following switches in `rebpp.py` are experiment controls rather than
independent algorithms: `VALIDINEQ2`, `BRANCH_AND_CUT`, `NO_VAR_GEN`,
`SOS_SOLVE`, `MIP_START_OR_HINT`, `DEBUG_CB`, `DEBUG_CB_0`, `DEBUG_CB_2`,
`DEBUG_INEQ_NOVAR`, and `ITEMSYMBREAK`. They are retained because they define
the paper's computational variants, but several are disabled in the default
configuration. They should not be removed without first checking whether a
reported experiment depends on them.

The module imports `pyscipopt` for historical compatibility, but the active
solver path uses Pyomo with Gurobi. If SCIP is no longer required by any
external experiment, that import and dependency are candidates for cleanup.

The source still contains commented historical paths and duplicate explanatory
comments from earlier experiments. They do not affect execution, but can be
consolidated in a future maintenance pass after the published experiments are
fully frozen.

### Static quality checks

Run the following before submitting a change:

```bash
.venv/bin/python -W error -m py_compile \
	src/knapsack.py src/sos2.py src/rebpp.py src/rbptest_grb.py
```

For a stricter academic-quality pipeline, add `ruff` for PEP 8/style checks,
`mypy` or `pyright` for static typing, and focused numerical regression tests
that compare the SOS2 objective with both DP variants on fixed seeds.
