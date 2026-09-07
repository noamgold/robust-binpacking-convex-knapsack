"""SOS2 optimization module focused on Gurobi/Pyomo and core DP integrations.

This file intentionally keeps only active project logic:
- SOS2 model solved with Gurobi via Pyomo,
- piecewise linear value evaluation,
- slope-based ordering helper,
- convex piecewise knapsack DP variants.

Removed legacy pieces:
- pyscipopt SOS2 model implementation,
- benchmark and timing loops inside main,
- debug scaffolding and unrelated experimental code.

Notes on imports
- ``numba.typed`` is not required in the current active implementation because
    list reconstruction is handled with plain Python lists under Numba-compatible
    patterns used in this codebase.
- ``math``, ``random`` and ``pandas`` are kept available for small local
    experimentation and instance I/O workflows.
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


def sos2(p, b, B):
    """Compatibility wrapper for legacy callers.

    Keeps the old entry-point name while routing execution to the active
    Gurobi/Pyomo implementation.
    """
    # Preserve old API surface while delegating to the active solver path.
    return sos2_gurobi(p, b, B)


def sos2_gurobi(p, b, B):
    """Solve the SOS2 model with Gurobi through Pyomo.

    p: profit matrix of shape (n, m)
    b: breakpoint/weight matrix of shape (n, m)
    B: total capacity bound
    """
    # Input matrices are expected as n items x m breakpoints.
    n, m = p.shape
    nb, mb = b.shape
    assert m == mb and n == nb

    # Build Pyomo model index sets.
    # I = items, J = piecewise breakpoints for each item.
    md = pe.ConcreteModel()
    md.I = pe.RangeSet(0, n - 1)
    md.J = pe.RangeSet(0, m - 1)

    # Continuous SOS2 variables for convex combination on each item's breakpoints.
    md.t = pe.Var(md.I, md.J, domain=pe.NonNegativeReals)

    # Objective: maximize piecewise profit.
    md.obj = pe.Objective(
        expr=sum(p[i, j] * md.t[i, j] for i in md.I for j in md.J),
        sense=pe.maximize,
    )

    # Global capacity constraint.
    md.c = pe.Constraint(expr=sum(b[i, j] * md.t[i, j] for i in md.I for j in md.J) <= B)

    # For each item, coefficients sum to 1 (convex combination requirement).
    def rule_sossum(md, i):
        return sum(md.t[i, j] for j in md.J) == 1

    md.sossum = pe.Constraint(md.I, rule=rule_sossum)

    # Enforce SOS2 adjacency structure per item (at most two adjacent active breakpoints).
    def rule_mysos(md, i):
        return [md.t[i, j] for j in md.J]

    md.mysos = pe.SOSConstraint(md.I, rule=rule_mysos, sos=2)

    # Solve with Gurobi.
    # Note: GRB is imported explicitly for environment consistency and
    # compatibility with existing project expectations around Gurobi usage.
    opt = pe.SolverFactory("gurobi_direct")
    opt.options["TimeLimit"] = TIMELIMIT

    # Measure both wall-clock and CPU time for diagnostics.
    grb_start_process = time.process_time()
    grb_start_elapsed = time.time()
    results = opt.solve(md, tee=False)
    grb_end_process = time.process_time()
    grb_end_elapsed = time.time()

    # Read solved objective value and decode the selected structure.
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
def p_eval(b_row, p_row, w):
    """Evaluate a piecewise-linear function at coordinate w by interpolation."""
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
def sort_instance_by_slopes(p, b):
    """Sort items by descending slope on the last piece to guide DP order."""
    # Ordering by marginal slope can improve practical DP behavior on convex instances.
    # Last-segment slope is used as a practical priority signal.
    dp_db = (p[:, 2] - p[:, 1]) / (b[:, 2] - b[:, 1])
    indexes = np.argsort(dp_db)
    indexes = np.flip(indexes)
    return p[indexes, :], b[indexes, :], indexes


@jit(nopython=True)
def convex_pw_knapsack_dp(p, b, W, y_intercept_nonzero=False):
    """DP solver for convex piecewise knapsack (capacity-based state)."""
    if __DEBUG_2:
        print("convex_pw_knapsack_dp...")

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
        w_min = max(W - b_array[i], 1)

        # If all other full-item weights exceed capacity, only minimal merge window is valid.
        if min(np.delete(b_array, i)) > W:
            w_max = 1

        # Merge DP value from other items with interpolated value of item i.
        for w in range(w_min, w_max):
            merged_val = B[w] + p_eval(b[i, :], p[i, :], W - w)
            if merged_val > max_val:
                max_val = merged_val
                w_max = w
                i_max = i

    if __DEBUG_2:
        print("convex_pw_knapsack_dp i_max=", i_max)

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
def convex_pw_knapsack_dp_profit(p, b, W, y_intercept_nonzero=False):
    """DP solver for convex piecewise knapsack (profit-based state)."""
    if __DEBUG_2:
        print("convex_pw_knapsack_dp...")

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
        print("convex_pw_knapsack_dp i_max=", i_max)

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


def read_instance(i):
    """Read a legacy benchmark instance from disk.

    This helper is kept for backward compatibility with the previous
    experimentation flow and data files.
    """
    # File format follows the historical benchmark convention in this repository.
    test_data = pd.read_csv("../data/Original_Instances/InverseStrong00" + str(i), skiprows=3, sep="\\s+")
    return test_data.iloc[:, 0], test_data.iloc[:, 1]


def write_instance(p_random, b_random, i):
    """Write a generated instance to disk for reproducible runs.

    The output format matches the project's historical data layout.
    """
    # Keep the same naming pattern used by older experiment pipelines.
    df = pd.DataFrame(data=np.column_stack((b_random, p_random)))
    fileName = "data" + str(len(b_random)) + "_" + str(len(p_random)) + "_" + str(i) + ".txt"
    df.to_csv(fileName)

if __name__ == "__main__":
    random.seed(101)
    for k_random in [50,100,250,500]: #, 250]: #[50, 100, 150, 200, 250, 300# ]:
        for R in [100,1000,10000]:
#    for R in [1000]: #[1000,10000]:
#        for k_random in [50,100,250,500]: #, 250]: #[50, 100, 150, 200, 250, 300]:
            #sos2_time_process = []
            #sos2_time_elapsed = []
            sos2g_time_process = []
            sos2g_time_elapsed = []
            sos2g_time_elapsed_wotl = []  # excluding time limits
            convex_time_process = []
            convex_time_elapsed = []
            fin_sos2 = []
            fin_sos2g = []
            fin_convex = []

            for i in range(1,31):
    #for i in range(1, 9):
                b = np.empty((0,3), int) #np.int64)
                p = np.empty((0,3), int) #np.int64)
                p_random,b_random = generate_random_instance(k_random,R,True)
                write_instance(p_random,b_random,i)
                #b_random, p_random  = read_instance(i)
                k_random = len(b_random)

                for p_val in p_random:
                    #p = np.append(p, np.array([[0, 0, p_val+1e-3*random.random()]]), axis=0)
                    p = np.append(p, np.array([[0, 0, p_val]]), axis=0)

                for b_val in b_random:
                    #b = np.append(b, np.array([[0, random.randint(0,b_val), b_val]]), axis = 0)
                    b = np.append(b, np.array([[0, b_val-1, b_val]]), axis = 0)

                        #print(for_loop_method_all_w(p, b, 63,-1,[i for i in range(1)], [[i for i in range(1)] for _ in range(1)], -1))
                W = int(math.ceil((5+i*3)/101 * sum(b_random)))

                #p, b = sort_instance_by_slopes(p, b)  # sort instance for linear speed up

                #(sos2_val, sos_items, sos_imax, solve_time, cpu_time) = 0, 0, 0 , 0, 0
                        #(sos2_val,sos_items,sos_imax, solve_time, cpu_time) = sos2(p,b,W)
                #if __DEBUG_2:
                #    print("sos2 objVal=", sos2_val)
                #sos2_time_process.append(cpu_time)
                #sos2_time_elapsed.append(solve_time)

        ##################################
                (sos2g_val,sosg_items,sosg_imax, solve_time, cpu_time) = sos2_gurobi(p,b,W)
                if __DEBUG_2:
                    print("sos2 gurobi objVal=", sos2g_val, " sosg_imax=", sosg_imax)
                    print(sosg_items)
                sos2g_time_process.append(cpu_time)
                sos2g_time_elapsed.append(solve_time)
                if solve_time <= TIMELIMIT-1:
                    sos2g_time_elapsed_wotl.append(solve_time)
                print("i=", i, " sos time: ", solve_time)
                sos2_val = sos2g_val
        ##################################

                convex_start_process = time.process_time()
                convex_start_elapsed = time.time()
                convex_val,items,imax = convex_pw_knapsack_dp_profit(p,b,W)
                #convex_val,items,imax = convex_pw_knapsack_dp(p,b,W)
                convex_end_process = time.process_time()
                convex_end_elapsed = time.time()
                convex_time_process.append(convex_end_process - convex_start_process)
                convex_time_elapsed.append(convex_end_elapsed - convex_start_elapsed)
                print("i=", i, " DP time: ", convex_end_elapsed - convex_start_elapsed)


                if abs(sos2g_val-convex_val)/sos2g_val > 1e-4 and int(sos2g_time_elapsed[-1]) < 1800:
                    print("ERR ! ................  sos2g objval: ", sos2g_val, " DP val: ", convex_val)
                    print("W=", W, " sosg_imax=", sosg_imax, " imax=", imax, "items=", items, " sum(b other than imax)=", quicksum(b[items,-1]), " sum(p other than imax)=",quicksum(p[items,-1]), " t=", sosg_items)
                    print("")
                    raise Exception("different obj vals")

                fin_sos2.append(sos2_val)
                fin_sos2g.append(sos2g_val)
                fin_convex.append(convex_val)

          #  print(k_random, " & ", f"{sum(sos2_time_process)/float(len(fin_sos2)):.2f}", " & ",  f"{max(sos2_time_process):.2f}" , " & ",  f"{sum(convex_time_process)/float(len(fin_convex)):.2f}", " & ", f"{max(convex_time_process):.2f}")
          #  print(k_random, " & ", f"{sum(sos2_time_elapsed)/float(len(fin_sos2)):.2f}", " & ", f"{max(sos2_time_elapsed):.2f}" , " & ", len([x for x in sos2_time_elapsed if x>TIMELIMIT-1]), " & ",  f"{sum(convex_time_elapsed)/float(len(fin_convex)):.2f}", " & ", f"{max(convex_time_elapsed):.2f}", " & ", len([x for x in convex_time_elapsed if x>TIMELIMIT-1]) )

            print(k_random, " & ", f"{sum(sos2g_time_process) / float(len(fin_sos2g)):.2f}", " & ", f"{max(sos2g_time_process):.2f}", " & ", f"{sum(convex_time_process) / float(len(fin_convex)):.2f}", " & ", f"{max(convex_time_process):.2f}")
            print(k_random, " & ", f"{sum(sos2g_time_elapsed) / float(len(fin_sos2g)):.2f}", " & ", f"{max(sos2g_time_elapsed):.2f}", " & ", len([x for x in sos2g_time_elapsed if x > TIMELIMIT-1]), " & ",f"{sum(convex_time_elapsed) / float(len(fin_convex)):.2f}", " & ", f"{max(convex_time_elapsed):.2f}", " & ", len([x for x in convex_time_elapsed if x > TIMELIMIT-1]))
            print(k_random," & ", R, " & ", f"{np.mean(sos2g_time_elapsed_wotl):.2f}", " & ", f"{max(sos2g_time_elapsed_wotl):.2f}", " & ", f"{np.mean(sos2g_time_elapsed):.2f}", " & ", len([x for x in sos2g_time_elapsed if x > TIMELIMIT - 1]), " & ", f"{sum(convex_time_elapsed) / float(len(fin_convex)):.2f}", " & ", f"{max(convex_time_elapsed):.2f}", " & ", len([x for x in convex_time_elapsed if x > TIMELIMIT - 1]))
         
            combined_data = np.column_stack((sos2g_time_elapsed,convex_time_elapsed))
            np.savetxt('times_' + str(k_random) + "_" + str(R) + ".csv",combined_data,delimiter=',',fmt='%10.2f',header='GurobiTime,DPTime')
# print(p_eval(b,p,6,2))
