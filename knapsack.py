"""Core knapsack DP utilities used by the SOS2 formulation.

This module intentionally keeps only the functions that are relevant to the
current project:
- upper-bound estimation,
- weight-based DP,
- profit-based DP,
- synthetic instance generation.

Legacy bin-packing, pyscipopt, and benchmark code were removed because they are
not part of the active optimization workflow.

Removed legacy functions and reasons:
- RandomBinPacking(): test-only random instance generator for bin packing; not used
  by the current knapsack/SOS2 solver workflow.
- BinPackingExample(): educational bin-packing example; unrelated to the active
  optimization problem and not part of the project scope.
- FFD(): First Fit Decreasing heuristic is a bin-packing heuristic, not an exact
  knapsack algorithm; removed to keep the module focused on exact DP methods.
- bpp(): pyscipopt-based bin-packing model; removed because the project requires
  a Gurobi/Pyomo-only workflow.
- solveBinPacking(): legacy experimental solver around the removed bin-packing
  formulation; removed as part of the cleanup.
- min_weight(), p_star(), f(), p_opt(): recursive or exploratory methods used in
  earlier experiments; they are not part of the current production logic and were
  removed to keep the codebase clear and maintainable.
- knapsack(): old pyscipopt knapsack solver; removed because the project targets
  Gurobi/Pyomo optimization and DP-based formulations instead.
"""

import math
import random
import sys
import time

import numpy as np
from numba import jit
from gurobipy import GRB
import pyomo.environ as pe

np.random.seed(0)


# compute upper bound based on relaxation and optional decreasing ratio ordering
@jit(nopython=True)
def P_upper_bound(w, p, W, indexes=[]):
    # curr_total_weight = total weight currently packed into the knapsack
    # p_bar = running upper-bound estimate of attainable profit
    curr_total_weight = 0
    p_bar = 0

    # n = number of items in the instance
    n = len(p)

    # Use the original arrays for readability. These are 1D vectors: one value per item.
    w_array = w
    p_array = p

    # If no explicit ordering was provided, compute rankings by profit/weight ratio.
    # A higher ratio means the item is more profitable per unit of used capacity.
    if len(indexes) < n:
        # Example: ratios[i] = p[i] / w[i]
        ratios = np.divide(p_array, w_array)

        # Sort indices from highest ratio to lowest ratio.
        indexes = np.argsort(ratios)

        # Reverse order so the best ratios are processed first.
        reversed_indexes = np.flip(indexes)
        indexes = reversed_indexes

    # Walk through the items in the chosen order.
    for index in indexes:
        # weight = how much capacity this item consumes
        # profit = how much value this item contributes
        weight = w_array[index]
        profit = p_array[index]

        # If the item fits fully in the remaining capacity, take it entirely.
        if curr_total_weight + weight < W:
            # Ignore zero-profit items; they do not improve the objective.
            if profit > 0:
                curr_total_weight += weight
                p_bar += profit

        # Otherwise, the item does not fit completely.
        # We take only a fraction of it, which is the standard fractional-knapsack relaxation.
        else:
            # remaining_space = how much capacity is still free
            remaining_space = W - curr_total_weight

            # fraction of the item that can fit in the remaining capacity
            fraction = remaining_space / weight

            # Add only the proportional profit for the usable fraction.
            p_bar += profit * fraction

            # Stop immediately: this is an upper bound, so once the knapsack is full,
            # taking additional items is no longer relevant for this relaxation.
            return int(math.floor(p_bar))

    # If all items were processed without exceeding capacity, return the final estimate.
    return int(math.floor(p_bar))


@jit(nopython=True)
def for_loop_method_all_w_save_all(p, w, W, B_all):
    # This DP table stores, for every capacity value, the best profit seen so far.
    # B_all[k, weight] = best profit after processing the first k+1 items and using exactly
    # 'weight' capacity.
    n = len(p)
    for k in range(0, n):
        # Copy the previous layer before updating with the new item.
        if k >= 1:
            B_all[k, :] = B_all[k - 1, :].copy()

        # Skip zero-profit items since they never help the objective.
        if p[k] > 0:
            # Try every possible capacity value from this item's weight onward.
            for weight in range(w[k], W + 1):
                if k == 0:
                    # Base case: first item alone takes the value p[0].
                    B_all[0, weight] = p[0]
                else:
                    # Standard knapsack transition:
                    # either skip the current item, or include it
                    # and add its profit to the best value at the smaller remaining capacity.
                    if B_all[k - 1, weight - w[k]] + p[k] > B_all[k - 1, weight]:
                        B_all[k, weight] = B_all[k - 1, weight - w[k]] + p[k]


@jit(nopython=True)
def for_loop_method_all_p_save_all(p, w, Pmax, B_all):
    # This version is profit-based, not capacity-based.
    # B_all[k, profit] = minimum weight needed to achieve exactly 'profit'
    # after processing the first k+1 items.
    n = len(p)
    B_all[0, 0] = 0

    for k in range(n):
        if k >= 1:
            B_all[k, :] = B_all[k - 1, :].copy()

        if p[k] > 0:
            if k == 0:
                # Base case: if we want profit p[0], the minimum weight needed is w[0].
                B_all[0, p[0]] = w[0]
            else:
                # Try every profit value from p[k] up to Pmax.
                for profit in range(p[k], Pmax + 1):
                    # If we can reach (profit - p[k]) using previous items,
                    # then adding item k gives a new candidate with weight + w[k].
                    if B_all[k - 1, profit - p[k]] < sys.maxsize and B_all[k - 1, profit - p[k]] + w[k] < B_all[k - 1, profit]:
                        B_all[k, profit] = B_all[k - 1, profit - p[k]] + w[k]


@jit(nopython=True)
def for_loop_method_all_w(p, w, W, B_in=np.array([]), i_skip=int(-1), return_items=True, sorted_two_piece=False):
    # This is the main capacity-based DP function.
    # It computes the best profit achievable for each total weight value from 0 to W.
    n = len(p)
    nn = len(w)

    # Safety check: the number of profits and weights must match.
    assert W >= 0
    if n != nn:
        raise Exception("n!=n")

    # B[weight] = best profit achievable using exactly 'weight' capacity.
    B = np.zeros(W + 1)

    # items[weight] stores which item indices were selected to achieve that profit.
    items = [[i for i in range(0)] for _ in range(W + 1)]

    # Only continue if there is at least one item whose weight is not larger than W.
    if np.min(w) <= W:
        # If a previous DP state was supplied, start from it instead of zero.
        if len(B_in) > 0 and i_skip >= 0:
            B = B_in.copy()

        # Determine the iteration range for the DP.
        i_max = n
        if sorted_two_piece:
            i_max = i_skip + 2

        # Process items one by one.
        for k in range(i_skip + 1, i_max):
            # Skip zero-profit items.
            if p[k] == 0:
                continue

            # A = previous DP state before item k is considered.
            A = B.copy()
            items_tmp = items.copy()

            # For each capacity value, check whether adding item k improves the solution.
            for weight in range(w[k], W + 1):
                # If the current item fits and improves the previous best, update it.
                if A[weight - w[k]] + p[k] > A[weight]:
                    # Update the best profit for this capacity.
                    B[weight] = A[weight - w[k]] + p[k]

                    # If requested, keep the actual selected items for reconstruction.
                    if return_items:
                        items[weight] = items_tmp[weight - w[k]] + [k]

    # Return the DP table and the selected-item reconstruction data.
    return B, items


def generate_random_instance(k_random, R, inversely_cor=True):
    # Generate synthetic knapsack data for benchmarking and testing.
    # Each item gets a profit and a weight.
    b_random = np.zeros(k_random, dtype=int)
    p_random = np.zeros(k_random, dtype=int)

    if inversely_cor:
        # Create a profit value for each item, then set its weight slightly larger.
        # This creates a strong inverse relationship between profit and weight.
        p_random = np.random.rand(k_random) * R
        p_random = np.ceil(p_random)
        for j in range(k_random):
            b_random[j] = p_random[j] + int(R / 10)
    else:
        # Alternative generation pattern for more irregular examples.
        b_random = np.random.rand(k_random) * (R - 1)
        b_random = 1 + (np.ceil(b_random)).astype(int)
        for j in range(k_random):
            p_random[j] = random.randint(
                math.floor(b_random[j] + R / 10 - R / 500),
                math.ceil(b_random[j] + R / 10 + R / 500),
            )

    return p_random, b_random


@jit(nopython=True)
def for_loop_method_all_p(p, w, P, B_in=np.array([], dtype=np.int64), i_skip=int(-1), return_items=True, sorted_two_piece=False):
    # Profit-based DP:
    # B[profit] = minimal total weight needed to achieve exactly `profit`.
    # This is dual to the weight-based DP where state is capacity.
    n = len(p)
    B = np.full(P + 1, sys.maxsize)
    items = [[i for i in range(0)] for _ in range(P + 1)]

    if len(B_in) > 0 and i_skip >= 0:
        # Optional warm-start from a previous DP layer/state.
        B = B_in.copy()

    i_max = n
    if sorted_two_piece:
        # Optional restricted scan used by two-piece optimization shortcuts.
        i_max = i_skip + 2

    B[0] = int(0)
    for k in range(i_skip + 1, i_max):
        # Skip non-contributing items.
        if p[k] == 0:
            continue
        A = B.copy()
        items_tmp = items.copy()

        for profit in range(p[k], P + 1):
            # Transition:
            # If profit-p[k] is reachable, then taking item k creates a candidate
            # solution for `profit` with added weight w[k].
            if A[profit - p[k]] < sys.maxsize and A[profit - p[k]] + w[k] < A[profit]:
                B[profit] = A[profit - p[k]] + w[k]
                if return_items:
                    items[profit] = items_tmp[profit-p[k]] + [k]
    #for p in range(P+1):
    #    if B[p] <= W:
    #        max_p = p

    return B, items #max_p


def for_loop_method_profit(P,p,w,W):
    # Legacy profit-state DP helper:
    # B[profit] stores the minimum weight required to realize that profit.
    n = len(p)
    B = [float('inf')] * (P + 1)
    B[0] = 0

    for k in range(n):
        # Snapshot previous stage to preserve 0/1 choice semantics per item.
        A = B.copy()

        for profit in range(p[k], P + 1):
            # Transition: include item k if it improves the minimum weight for `profit`.
            if A[profit - p[k]] + w[k] < A[profit]:
                B[profit] = A[profit - p[k]] + w[k]

    # Extract the largest profit that fits within capacity W.
    for p in range(P+1):
        if B[p] <= W:
            max_p = p

    return max_p


def knapsack(w, p, W):
    """Compatibility helper used by the legacy benchmark main block.

    Returns the best DP value at capacity W using the active weight-based DP
    implementation.
    """
    # Route legacy `knapsack(...)` call-sites to the active DP implementation.
    B, _ = for_loop_method_all_w(np.array(p), np.array(w), int(W), np.array([], dtype=float), int(-1), False)
    return B[int(W)]



if __name__ == "__main__":

    """
    Code below used to track run times for the for loop method vs the scip method
    and can also be used to track run times for the f method vs g method (profit vs weight)
    """

    # Benchmark harness kept for compatibility with existing local workflows.
    st = time.time()

    fin_g = []
    fin_f = []
    fin_knap = []
    fin_for = []
    g_time_process = []
    f_time_process = []
    knap_time_process = []
    for_time_process = []
    g_time_elapsed = []
    f_time_elapsed = []
    knap_time_elapsed = []
    for_time_elapsed = []
    bounds = []

    # Repeat randomized experiments to average runtime noise.
    for i in range(10): # number of trials to average out on
        print(i) # to see what iteration it's on
        R = 100
        k_random = 50 #500 # number of items
        n = k_random
        w_random = []
        

        p_random = np.random.rand(k_random)
        p_random *= R
        p_random = np.array(np.rint(p_random), dtype='i')

        

        for j in range(k_random):
            adjusted_weight_j = p_random[j] + int(R/10)
            w_random.append(adjusted_weight_j)
        

        W_random = int(20/101 * sum(w_random))
        # Fast relaxation bound used for diagnostics and optional downstream limits.
        P_max = P_upper_bound(w_random, p_random, W_random)
        bounds.append(P_max)


        knap_start_process = time.process_time()
        knap_start_elapsed = time.time()
        # DP-based compatibility solver (replaces old pyscipopt implementation).
        knap_star = knapsack(w_random, p_random, W_random)
        knap_stop_process = time.process_time()
        knap_stop_elapsed = time.time()
        knap_time_process.append(knap_stop_process - knap_start_process)
        knap_time_elapsed.append(knap_stop_elapsed - knap_start_elapsed)
        print("done knap")

        # f_start_process = time.process_time()
        # f_start_elapsed = time.time()
        # # #f_star = p_opt_trial(P_max, k_random, w_random, p_random, W_random)
        # f_star = p_opt(P_max, k_random, w_random, p_random, W_random)
        # f_stop_process = time.process_time()
        # f_stop_elapsed = time.time()
        # f_time_process.append(f_stop_process - f_start_process)
        # f_time_elapsed.append(f_stop_elapsed - f_start_elapsed)
        # print("done f")

        # g_start_process = time.process_time()
        # g_start_elapsed = time.time()
        # g_star = g(p_random,w_random,W_random,k_random)
        # g_stop_process = time.process_time()
        # g_stop_elapsed = time.time()
        # g_time_process.append(g_stop_process - g_start_process)
        # g_time_elapsed.append(g_stop_elapsed - g_start_elapsed)
        # print("done g")

        for_start_process = time.process_time()
        for_start_elapsed = time.time()
        # Direct DP call kept for side-by-side timing with the compatibility wrapper.
        for_star = for_loop_method_all_w(np.array(p_random), np.array(w_random), W_random, np.array([],dtype=float), #[List().append(-1) for _ in range(1)],
                                         int(-1))
        for_stop_process = time.process_time()
        for_stop_elapsed = time.time()
        for_time_process.append(for_stop_process - for_start_process)
        for_time_elapsed.append(for_stop_elapsed - for_start_elapsed)
        print("done with for loop method")


        # fin_f.append(f_star)
        # fin_g.append(g_star)
        fin_knap.append(knap_star)
        fin_for.append(for_star)


    et = time.time()
    # print(fin_g)
    # print(fin_f)
    print(fin_knap)
    print(fin_for)
    print("Bounds: ", bounds)

    # Final aggregate summary for quick regression checks.
    print("The average processing time for knap and the for loop method respectively are: ", sum(knap_time_process)/float(len(fin_knap)), sum(for_time_process)/float(len(fin_for)))
    print("The average elapsed time for knap and the for loop method respectively are: ", sum(knap_time_elapsed)/float(len(fin_knap)), sum(for_time_elapsed)/float(len(fin_for)))
    # print("The average processing time for f, g, and knap respectively are: ",sum(f_time_process)/float(len(fin_f)), sum(g_time_process)/float(len(fin_g)), sum(knap_time_process)/float(len(fin_knap)))
    # print("The average elapsed time for f, g, and knap respectively are: ",sum(f_time_elapsed)/float(len(fin_f)), sum(g_time_elapsed)/float(len(fin_g)), sum(knap_time_elapsed)/float(len(fin_knap)))
    # print(for_loop_method_profit(P,p,w,W))
