"""SOS2 and dynamic-programming solvers for the two-piece convex knapsack.

For paper notation, ``p[i, :]`` contains the breakpoint profits and ``b[i, :]``
contains the corresponding weights for item/bin ``i``.  ``B`` is the global
capacity ``Omega``.  ``sos2_gurobi`` implements formulation (16): ``t[i, j]``
are convex-combination variables and the SOS2 constraint enforces adjacency.
The DP routines implement Algorithm 2 and Appendix A, returning the value
``P*`` of the two-piece convex knapsack separation problem.
"""

from json.encoder import INFINITY

import numpy as np
from numba import jit
import time
import pyomo.environ as pe
from gurobipy import GRB
import sys
import math
import random
import pandas as pd
from typing import List, Tuple

# Keep quicksum-based debug print lines compatible with the historical main block.
quicksum = np.sum

from knapsack import (
    for_loop_method_all_w,
    for_loop_method_all_p,
    for_loop_method_all_w_save_all,
    for_loop_method_all_p_save_all,
    P_upper_bound,
    generate_random_instance,
)

__DEBUG_2 = False
NZ_TOL = 1e-6
TIMELIMIT = 1800


def sos2(p: np.ndarray, b: np.ndarray, B: int) -> Tuple[float, List[int], int, float, float]:
    """Solve the 2PCK separation problem through the SOS2 formulation.

    Parameters
    ----------
    p, b : numpy.ndarray
        Breakpoint profits and weights for the convex functions
        ``p_j(x_j) = (gamma_j + beta_j x_j)_+``.
    B : int
        Global uncertainty budget ``Omega``.

    Returns
    -------
    tuple
        Objective value ``P*``, selected full items, pivot item, elapsed wall
        time, and elapsed CPU time.

    Notes
    -----
    This is a compatibility entry point for the paper's SOS2 benchmark; the
    implementation delegates to :func:`sos2_gurobi`.
    """

    # Preserve old API surface while delegating to the active solver path.
    return sos2_gurobi(p, b, B)


def sos2_gurobi(
    p: np.ndarray, b: np.ndarray, B: int
) -> Tuple[float, List[int], int, float, float]:
    """Solve the SOS2 model with Gurobi through Pyomo.

    ``p``: profit matrix of shape ``(n, m)``.
    ``b``: breakpoint/weight matrix of shape ``(n, m)``.
    ``B``: total capacity bound, corresponding to ``Omega``.
    """
    # Input matrices are expected as n items x m breakpoints.
    n, m = p.shape
    nb, mb = b.shape
    assert m == mb and n == nb

    # Formulation (16): I indexes convex functions and J indexes breakpoints.
    md = pe.ConcreteModel()
    md.I = pe.RangeSet(0, n - 1)
    md.J = pe.RangeSet(0, m - 1)

    md.t = pe.Var(md.I, md.J, domain=pe.NonNegativeReals)

    # Eq. (16a): maximize the convex-combination representation of total profit.
    md.obj = pe.Objective(
        expr=sum(p[i, j] * md.t[i, j] for i in md.I for j in md.J),
        sense=pe.maximize,
    )

    # Eq. (16b): enforce the global uncertainty budget Omega.
    md.c = pe.Constraint(expr=sum(b[i, j] * md.t[i, j] for i in md.I for j in md.J) <= B)

    def rule_sossum(md, i):
        # Eq. (16c): each function is represented by a convex combination.
        return sum(md.t[i, j] for j in md.J) == 1

    md.sossum = pe.Constraint(md.I, rule=rule_sossum)

    def rule_mysos(md, i):
        # Eq. (16d): restrict support to adjacent breakpoints.
        return [md.t[i, j] for j in md.J]

    md.mysos = pe.SOSConstraint(md.I, rule=rule_mysos, sos=2)

    # Solve formulation (16) with Gurobi's SOS2 branching implementation.
    opt = pe.SolverFactory("gurobi_direct")
    opt.options["TimeLimit"] = TIMELIMIT

    grb_start_process = time.process_time()
    grb_start_elapsed = time.time()
    results = opt.solve(md, tee=False)
    grb_end_process = time.process_time()
    grb_end_elapsed = time.time()

    objVal = md.obj()
    fin = []
    i_max = -1

    if "ok" == str(results.Solver.status):
        # Recover selected items and detect the item with fractional middle selection.
        # Interpretation:
        # - j == m-1 with positive value => item considered "fully selected" in this encoding.
        # - 0 < t[i,j] < 1 for an interior j can indicate the pivot/fractional item.
        for i in md.I:
            if p[i, m - 1] > 0:
                for j in md.J:
                    val = md.t[i, j]()
                    if val > NZ_TOL:
                        if j < m - 1 and val < 1 - NZ_TOL:
                            i_max = i
                            break
                        elif j == m - 1:
                            fin.append(i)
    else:
        print("No Valid Solution Found")
        objVal = -INFINITY

    return (
        objVal,
        fin,
        i_max,
        grb_end_elapsed - grb_start_elapsed,
        grb_end_process - grb_start_process,
    )


@jit(nopython=True)
def p_eval(b_row: np.ndarray, p_row: np.ndarray, w: float) -> float:
    r"""Evaluate ``p_j(x_j)`` by linear interpolation at ``x_j = w``.

    Parameters
    ----------
    b_row, p_row : numpy.ndarray
        Ordered breakpoints and values for one convex function.
    w : float
        Allocated uncertainty budget for the function.

    Returns
    -------
    float
        Interpolated value ``\hat p_j(w)``.
    """
    # b_row: sorted x breakpoints, p_row: corresponding y values.
    b_max = b_row[-1]
    if w < 0:
        return -sys.maxsize

    assert 0 <= w <= b_max
    if w == 0:
        return p_row[0]

    # searchsorted returns the first index whose breakpoint is >= w.
    right_index = np.searchsorted(b_row, w)
    if b_row[right_index] == w:
        return p_row[right_index]

    # Interpolate on the segment [left_index, right_index].
    left_index = right_index - 1
    assert b_row[right_index] > b_row[left_index]
    difference = float(w - b_row[left_index])
    fraction = difference / float(b_row[right_index] - b_row[left_index])
    return (1 - fraction) * p_row[left_index] + fraction * p_row[right_index]


@jit(cache=True, nopython=True)
def sort_instance_by_slopes(
    p: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Sort 2PCK functions by the slope order required by Eq. (12).

    This is the ordering assumed by Eq. (12) and Algorithm 2.
    Parameters
    ----------
    p, b : numpy.ndarray
        Profit and weight breakpoints for the 2PCK instance.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Sorted matrices and the permutation mapping sorted to original rows.
    """
    # Ordering by marginal slope can improve practical DP behavior on convex instances.
    # Last-segment slope is used as a practical priority signal.
    dp_db = (p[:, 2] - p[:, 1]) / (b[:, 2] - b[:, 1])
    indexes = np.argsort(dp_db)
    indexes = np.flip(indexes)
    return p[indexes, :], b[indexes, :], indexes


@jit(nopython=True)
def convex_pw_knapsack_omega_dp(
    p: np.ndarray,
    b: np.ndarray,
    W: int,
    y_intercept_nonzero: bool = False,
) -> Tuple[float, np.ndarray, int]:
    r"""Solve 2PCK with the weight-indexed Omega-DP from Appendix A.

    This is the Omega-DP variant described in Appendix A, Eq. (18).
    Parameters
    ----------
    p, b : numpy.ndarray
        Three-breakpoint representation of ``\hat p_j``.
    W : int
        Uncertainty budget ``Omega``.
    y_intercept_nonzero : bool, optional
        Whether fixed intercepts should be separated from the DP objective.

    Returns
    -------
    tuple[float, numpy.ndarray, int]
        Optimal value, selected full-item indices, and fractional pivot index.
    """
    if __DEBUG_2:
        print("convex_pw_knapsack_omega_dp...")

    n, m = p.shape
    nb, mb = b.shape
    assert m == mb and n == nb

    w_max = 0
    i_max = int(-1)
    p, b, origIdxs = sort_instance_by_slopes(p, b)

    # initialP collects fixed intercept profit when modeling non-zero y-intercepts.
    initialP = 0
    if y_intercept_nonzero:
        for i in range(n):
            zeroIdxs = np.where(b[i, :] == 0)[0]
            lastZero = max(zeroIdxs)
            initialP += p[i, lastZero]
            # Shift piece values so the DP merge is performed on incremental gain only.
            p[i, lastZero : m - 1] = p[i, lastZero : m - 1] - p[i, lastZero]

    # Last breakpoint of each item acts like "full-item" proxy for the DP backbone.
    profit_array = p[:, m - 1]
    b_array = b[:, m - 1]
    max_val = float(0)

    B_all = np.zeros((n, W + 1), dtype=float)
    B = np.zeros(W + 1)

    # Iterate candidate "special" item i in reverse order.
    for i in range(n - 1, -1, -1):
        # Temporarily exclude item i from the backbone DP, then merge it through p_eval.
        pi = profit_array[i]
        profit_array[i] = 0
        skip_idx = i

        if i == 0 or i == n - 1:
            skip_idx = -1

        # First iteration builds a full baseline table, later iterations re-use
        # partial state through skip-aware DP calls.
        if i == n - 1:
            for_loop_method_all_w_save_all(profit_array, b_array, W, B_all)
            B = B_all[n - 1, :].copy()
        else:
            B, _ = for_loop_method_all_w(
                profit_array,
                b_array,
                W,
                B_all[max(skip_idx - 1, -1), :],
                skip_idx,
                False,
                True,
            )

        profit_array[i] = pi
        w_max = W
        w_min = max(W - b_array[i], 0)

        # Appendix A maximizes over U in [0, Omega], including both endpoints.
        # U = 0 is required when the pivot receives the full uncertainty budget.
        for w in range(w_min, w_max + 1):
            merged_val = B[w] + p_eval(b[i, :], p[i, :], W - w)
            if merged_val > max_val:
                max_val = merged_val
                w_max = w
                i_max = i

    if __DEBUG_2:
        print("convex_pw_knapsack_omega_dp i_max=", i_max)

    # Reconstruct selected non-pivot items from the final DP pass.
    p_vec = profit_array.copy()
    b_vec = b_array.copy()
    if i_max != -1:
        p_vec[int(i_max)] = 0
        b_vec[int(i_max)] = W

    _, items_max = for_loop_method_all_w(p_vec, b_vec, W)

    retIdxs = np.array(items_max[w_max])
    retImax = int(-1)
    if i_max != -1:
        retImax = origIdxs[i_max]

    return initialP + max_val, origIdxs[retIdxs], retImax


@jit(nopython=True)
def convex_pw_knapsack_profit_dp(
    p: np.ndarray,
    b: np.ndarray,
    W: int,
    y_intercept_nonzero: bool = False,
) -> Tuple[float, np.ndarray, int]:
    r"""Solve 2PCK with the profit-indexed DP of Algorithm 2.

    This is Algorithm 2, using the ``zeta`` tables from Eq. (10) and the
    one-fractional-item representation from Observation 2.
    Parameters
    ----------
    p, b : numpy.ndarray
        Three-breakpoint representation of ``\hat p_j``.
    W : int
        Uncertainty budget ``Omega``.
    y_intercept_nonzero : bool, optional
        Whether fixed intercepts should be separated from the DP objective.

    Returns
    -------
    tuple[float, numpy.ndarray, int]
        Optimal value, selected full-item indices, and fractional pivot index.
    """
    if __DEBUG_2:
        print("convex_pw_knapsack_profit_dp...")

    n, m = p.shape
    nb, mb = b.shape
    assert m == mb and n == nb

    p, b, origIdxs = sort_instance_by_slopes(p, b)
    i_max = int(-1)

    # Same intercept-handling idea as in capacity-state variant.
    initialP = 0
    if y_intercept_nonzero:
        for i in range(n):
            zeroIdxs = np.where(b[i, :] == 0)[0]
            lastZero = max(zeroIdxs)
            initialP += p[i, lastZero]
            p[i, lastZero : m - 1] = p[i, lastZero : m - 1] - p[i, lastZero]

    profit_array = p[:, m - 1].astype(np.int64)
    b_array = b[:, m - 1]

    # Ratio ordering is reused only to build a stable upper bound for profit-state DP width.
    ratios = np.divide(profit_array, b_array)
    indexes = np.argsort(ratios)
    reversed_indexes = np.flip(indexes)
    # Upper bound on profit state-space width; keeps the DP table finite and practical.
    Pmax = P_upper_bound(b_array, profit_array, W, reversed_indexes)

    max_val = float(0)
    B_all = np.full((n, Pmax + 1), sys.maxsize)
    B = np.full(Pmax + 1, sys.maxsize)

    # p_best stores the best profit-state index found in the merge loop.
    p_best = 0
    # Profit-state variant: keep minimal weight per profit, then merge with
    # piecewise value of one distinguished item.
    for i in range(n - 1, -1, -1):
        pi = profit_array[i]
        profit_array[i] = 0
        skip_idx = i
        # Recompute bound after temporary exclusion of candidate pivot item i.
        Pmaxi = P_upper_bound(b_array, profit_array, W, reversed_indexes)

        if i == 0 or i == n - 1:
            skip_idx = -1

        if i == n - 1:
            for_loop_method_all_p_save_all(profit_array, b_array, Pmax, B_all)
            B = B_all[n - 1, :].copy()
        else:
            B, _ = for_loop_method_all_p(
                profit_array,
                b_array,
                Pmaxi,
                B_all[max(skip_idx - 1, -1), :],
                skip_idx,
                False,
                True,
            )

        profit_array[i] = pi
        p_max = Pmax
        p_min = 1

        if min(np.delete(b_array, i)) > W:
            p_max = 1

        # Enumerate candidate profit states and test whether residual capacity
        # can host the distinguished piecewise item.
        for pp in range(p_min, p_max + 1):
            # Feasibility filters:
            # 1) backbone solution must fit capacity,
            # 2) residual capacity must not exceed pivot's terminal breakpoint.
            if B[pp] > W or W - B[pp] > b_array[i]:
                continue
            merged_val = float(pp) + p_eval(b[i, :], p[i, :], W - B[pp])
            if merged_val > max_val:
                max_val = merged_val
                p_best = pp
                i_max = i

    if __DEBUG_2:
        print("convex_pw_knapsack_profit_dp i_max=", i_max)

    # Final reconstruction of non-pivot items using profit-state DP.
    p_vec = profit_array.copy()
    b_vec = b_array.copy()
    if i_max != -1:
        p_vec[int(i_max)] = 0
        b_vec[int(i_max)] = W + 1

    _, items_max = for_loop_method_all_p(p_vec, b_vec, p_best)

    retIdxs = np.array(items_max[p_best])
    retImax = int(-1)
    if i_max != -1:
        retImax = origIdxs[i_max]

    return initialP + max_val, origIdxs[retIdxs], retImax


def read_instance(i: int) -> Tuple[np.ndarray, np.ndarray]:
    """Read a legacy benchmark instance from disk.

    This helper is kept for backward compatibility with the previous
    experimentation flow and data files.
    """
    # File format follows the historical benchmark convention in this repository.
    test_data = pd.read_csv("../data/Original_Instances/InverseStrong00" + str(i), skiprows=3, sep="\\s+")
    return test_data.iloc[:, 0], test_data.iloc[:, 1]


def write_instance(p_random: np.ndarray, b_random: np.ndarray, i: int) -> str:
    """Write a generated instance to disk and return its filename.

    The output format matches the project's historical data layout.
    """
    # Keep the same naming pattern used by older experiment pipelines.
    df = pd.DataFrame(data=np.column_stack((b_random, p_random)))
    fileName = "data" + str(len(b_random)) + "_" + str(len(p_random)) + "_" + str(i) + ".txt"
    df.to_csv(fileName)
    return fileName


if __name__ == "__main__":
    random.seed(101)
    for k_random in [50, 100, 250, 500]:
        for R in [100, 1000, 10000]:
            sos2g_time_process = []
            sos2g_time_elapsed = []
            sos2g_time_elapsed_wotl = []
            profit_time_process = []
            profit_time_elapsed = []
            omega_time_process = []
            omega_time_elapsed = []
            fin_sos2g = []
            fin_profit = []
            fin_omega = []

            for i in range(1, 31):
                b = np.empty((0, 3), int)
                p = np.empty((0, 3), int)
                p_random, b_random = generate_random_instance(k_random, R, True)
                write_instance(p_random, b_random, i)

                for p_val in p_random:
                    p = np.append(p, np.array([[0, 0, p_val]]), axis=0)
                for b_val in b_random:
                    b = np.append(b, np.array([[0, b_val - 1, b_val]]), axis=0)

                W = int(math.ceil((5 + i * 3) / 101 * sum(b_random)))

                sos2g_val, sosg_items, sosg_imax, solve_time, cpu_time = sos2_gurobi(p, b, W)
                sos2g_time_process.append(cpu_time)
                sos2g_time_elapsed.append(solve_time)
                if solve_time <= TIMELIMIT - 1:
                    sos2g_time_elapsed_wotl.append(solve_time)
                print("i=", i, " sos time: ", solve_time)

                profit_start_process = time.process_time()
                profit_start_elapsed = time.time()
                profit_val, profit_items, profit_imax = convex_pw_knapsack_profit_dp(p, b, W)
                profit_time_process.append(time.process_time() - profit_start_process)
                profit_time_elapsed.append(time.time() - profit_start_elapsed)
                print("i=", i, " Profit-DP time: ", profit_time_elapsed[-1])

                omega_start_process = time.process_time()
                omega_start_elapsed = time.time()
                omega_val, omega_items, omega_imax = convex_pw_knapsack_omega_dp(p, b, W)
                omega_time_process.append(time.process_time() - omega_start_process)
                omega_time_elapsed.append(time.time() - omega_start_elapsed)
                print("i=", i, " Omega-DP time: ", omega_time_elapsed[-1])

                if int(solve_time) < TIMELIMIT:
                    if abs(sos2g_val - profit_val) / sos2g_val > 1e-4:
                        raise ValueError("Profit-DP objective differs from SOS2")
                    if abs(sos2g_val - omega_val) / sos2g_val > 1e-4:
                        raise ValueError("Omega-DP objective differs from SOS2")

                fin_sos2g.append(sos2g_val)
                fin_profit.append(profit_val)
                fin_omega.append(omega_val)

            print(
                k_random, " & ", f"{sum(sos2g_time_process) / len(fin_sos2g):.2f}", " & ",
                f"{max(sos2g_time_process):.2f}", " & ",
                f"{sum(profit_time_process) / len(fin_profit):.2f}", " & ",
                f"{max(profit_time_process):.2f}", " & ",
                f"{sum(omega_time_process) / len(fin_omega):.2f}", " & ",
                f"{max(omega_time_process):.2f}",
            )
            print(
                k_random, " & ", R, " & ", f"{np.mean(sos2g_time_elapsed_wotl):.2f}", " & ",
                f"{max(sos2g_time_elapsed_wotl):.2f}", " & ", f"{np.mean(sos2g_time_elapsed):.2f}", " & ",
                len([value for value in sos2g_time_elapsed if value > TIMELIMIT - 1]), " & ",
                f"{np.mean(profit_time_elapsed):.2f}", " & ", f"{max(profit_time_elapsed):.2f}", " & ",
                len([value for value in profit_time_elapsed if value > TIMELIMIT - 1]), " & ",
                f"{np.mean(omega_time_elapsed):.2f}", " & ", f"{max(omega_time_elapsed):.2f}", " & ",
                len([value for value in omega_time_elapsed if value > TIMELIMIT - 1]),
            )

            combined_data = np.column_stack((sos2g_time_elapsed, profit_time_elapsed, omega_time_elapsed))
            np.savetxt(
                "times_" + str(k_random) + "_" + str(R) + ".csv",
                combined_data,
                delimiter=',',
                fmt='%10.2f',
                header='GurobiTime,ProfitDPTime,OmegaDPTime',
            )
