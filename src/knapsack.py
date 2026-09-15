"""Binary-knapsack dynamic programs used by the convex-knapsack algorithms.

Notation from the paper:
    ``w[i]`` is the integer weight ``u_i``; ``p[i]`` is the profit
    ``\bar p_i``; and ``W`` is the capacity ``Omega`` for weight-indexed DP
    or ``P``/``Pmax`` for profit-indexed DP.  The routines returning ``B``
    implement the recurrences for ``Pi_f(U,k)`` or ``zeta_f(P,k)`` after the
    excluded item and breakpoint convention have been encoded by the caller.

The module supplies the pseudo-polynomial building blocks used by Algorithm 2
and Appendix A of the paper. It does not solve the REBP master problem.
"""

import math
import random
import sys
import time
from typing import List, Tuple

import numpy as np
from numba import jit
from gurobipy import GRB
import pyomo.environ as pe

np.random.seed(0)


@jit(nopython=True)
def P_upper_bound(
    w: np.ndarray,
    p: np.ndarray,
    W: int,
    indexes: np.ndarray = np.array([], dtype=np.int64),
) -> int:
    r"""Compute a fractional-knapsack upper bound.

    Here ``w`` and ``p`` are the binary-knapsack weights and profits and ``W``
    is the capacity.  The bound is used to limit the profit-indexed state
    space ``Pmax`` in the convex-knapsack DP.
    Parameters
    ----------
    w, p : numpy.ndarray
        Binary-knapsack weights ``u`` and profits ``\bar p``.
    W : int
        Capacity, corresponding to ``Omega`` in the weight-indexed relaxation.
    indexes : numpy.ndarray, optional
        Item order. If empty, items are sorted by decreasing ``p[i] / w[i]``.

    Returns
    -------
    int
        An upper bound on the attainable binary-knapsack profit.

    Notes
    -----
    This bound determines a finite ``Pmax`` for the profit-indexed DP in
    Algorithm 2 of the paper. The typed empty-array default avoids a Numba
    fingerprinting failure caused by a Python list default.
    """
    # curr_total_weight = total weight currently packed into the knapsack
    # p_bar = running upper-bound estimate of attainable profit
    curr_total_weight = 0
    p_bar = 0

    # n = number of items in the instance
    n = len(p)

    # Use the original arrays for readability. These are 1D vectors: one value per item.
    w_array = w  # np.array(w)
    p_array = p  # np.array(p)

    # The fractional relaxation supplies a finite upper bound for P_max.
    if len(indexes) < n:
        ratios = np.divide(p_array, w_array)

        indexes = np.argsort(ratios)

        reversed_indexes = np.flip(indexes)
        indexes = reversed_indexes

    for index in indexes:
        weight = w_array[index]
        profit = p_array[index]

        if curr_total_weight + weight < W:
            if profit > 0:
                curr_total_weight += weight
                p_bar += profit

        else:
            remaining_space = W - curr_total_weight
            fraction = remaining_space / weight
            p_bar += profit * fraction
            return int(math.floor(p_bar))

    return int(math.floor(p_bar))


@jit(nopython=True)
def for_loop_method_all_w_save_all(
    p: np.ndarray, w: np.ndarray, W: int, B_all: np.ndarray
) -> None:
    r"""Build all prefix states of the weight-indexed DP.

    The table stores ``Pi_f(U,k)``, the maximum profit achievable with weight
    at most ``U`` using the first ``k`` items. The table is mutated in place.
    """
    n = len(p)
    # Populate all prefix states of the capacity-indexed recurrence Pi_f(U, k).
    for k in range(0, n):
        if k >= 1:
            B_all[k, :] = B_all[k - 1, :].copy()

        if p[k] > 0:
            for weight in range(w[k], W + 1):
                if k == 0:
                    B_all[0, weight] = p[0]
                else:
                    if B_all[k - 1, weight - w[k]] + p[k] > B_all[k - 1, weight]:
                        B_all[k, weight] = B_all[k - 1, weight - w[k]] + p[k]


@jit(nopython=True)
def for_loop_method_all_p_save_all(
    p: np.ndarray, w: np.ndarray, Pmax: int, B_all: np.ndarray
) -> None:
    r"""Build all prefix states of the profit-indexed DP.

    The table stores ``zeta_f(P,k)``, the minimum weight required to attain
    profit ``P`` with the first ``k`` items. The table is mutated in place.
    """
    n = len(p)
    # Populate all prefix states of the profit-indexed recurrence zeta_f(P, k).
    B_all[0, 0] = 0

    for k in range(n):
        if k >= 1:
            B_all[k, :] = B_all[k - 1, :].copy()

        if p[k] > 0:
            if k == 0:
                B_all[0, p[0]] = w[0]
            else:
                for profit in range(p[k], Pmax + 1):
                    if B_all[k - 1, profit - p[k]] < sys.maxsize and B_all[k - 1, profit - p[k]] + w[k] < B_all[k - 1, profit]:
                        B_all[k, profit] = B_all[k - 1, profit - p[k]] + w[k]


@jit(nopython=True)
def for_loop_method_all_w(
    p: np.ndarray,
    w: np.ndarray,
    W: int,
    B_in: np.ndarray = np.array([]),
    i_skip: int = int(-1),
    return_items: bool = True,
    sorted_two_piece: bool = False,
) -> Tuple[np.ndarray, List[List[int]]]:
    r"""Solve the capacity-indexed binary-knapsack recurrence.

    The returned value is the implementation of ``Pi_f(W,k)`` from Appendix A.
    Parameters
    ----------
    p, w : numpy.ndarray
        Profits ``\bar p`` and weights ``u``.
    W : int
        Capacity, usually ``Omega``.
    B_in : numpy.ndarray, optional
        Previously computed DP state for reuse across pivot exclusions.
    i_skip : int, optional
        Pivot item ``f`` excluded from the binary backbone.
    return_items : bool, optional
        Whether to reconstruct selected item indices.
    sorted_two_piece : bool, optional
        Restrict the scan to the ordered two-piece shortcut.

    Returns
    -------
    tuple[numpy.ndarray, list[list[int]]]
        Profit table and item reconstructions for each capacity.
    """
    n = len(p)
    nn = len(w)

    assert W >= 0
    if n != nn:
        raise Exception("n!=n")

    B = np.zeros(W + 1)

    items = [[i for i in range(0)] for _ in range(W + 1)]

    # Reuse the backbone table when evaluating an excluded pivot item f.
    if np.min(w) <= W:
        if len(B_in) > 0 and i_skip >= 0:
            B = B_in.copy()

        i_max = n
        if sorted_two_piece:
            i_max = i_skip + 2

        for k in range(i_skip + 1, i_max):
            if p[k] == 0:
                continue

            A = B.copy()
            items_tmp = items.copy()

            # The 0/1 transition reads the previous layer before updating B.
            for weight in range(w[k], W + 1):
                if A[weight - w[k]] + p[k] > A[weight]:
                    B[weight] = A[weight - w[k]] + p[k]

                    if return_items:
                        items[weight] = items_tmp[weight - w[k]] + [k]

    return B, items


def generate_random_instance(
    k_random: int, R: int, inversely_cor: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate an inverse-correlated knapsack benchmark instance.

    Parameters
    ----------
    k_random : int
        Number of items ``n``.
    R : int
        Upper scale of generated profits.
    inversely_cor : bool, optional
        If true, use the hard inverse-correlation construction from the paper.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Profit vector ``\bar p`` and weight vector ``u``.
    """
    b_random = np.zeros(k_random, dtype=int)
    p_random = np.zeros(k_random, dtype=int)

    if inversely_cor:
        p_random = np.random.rand(k_random) * R
        p_random = np.ceil(p_random)
        for j in range(k_random):
            b_random[j] = p_random[j] + int(R / 10)
    else:
        b_random = np.random.rand(k_random) * (R - 1)
        b_random = 1 + (np.ceil(b_random)).astype(int)
        for j in range(k_random):
            p_random[j] = random.randint(
                math.floor(b_random[j] + R / 10 - R / 500),
                math.ceil(b_random[j] + R / 10 + R / 500),
            )

    return p_random, b_random


@jit(nopython=True)
def for_loop_method_all_p(
    p: np.ndarray,
    w: np.ndarray,
    P: int,
    B_in: np.ndarray = np.array([], dtype=np.int64),
    i_skip: int = int(-1),
    return_items: bool = True,
    sorted_two_piece: bool = False,
) -> Tuple[np.ndarray, List[List[int]]]:
    r"""Solve the profit-indexed binary-knapsack recurrence.

    This is the implementation of the ``zeta_f(P,k)`` recurrence in Eq. (10).
    Parameters
    ----------
    p, w : numpy.ndarray
        Profits ``\bar p`` and weights ``u``.
    P : int
        Maximum profit state ``Pmax``.
    B_in : numpy.ndarray, optional
        Previously computed state for DP reuse.
    i_skip : int, optional
        Pivot item ``f`` excluded from the state.
    return_items, sorted_two_piece : bool, optional
        Reconstruction and ordered-shortcut controls.

    Returns
    -------
    tuple[numpy.ndarray, list[list[int]]]
        Minimum-weight table ``zeta_f(P,k)`` and reconstructions.
    """
    n = len(p)
    B = np.full(P + 1, sys.maxsize)
    items = [[i for i in range(0)] for _ in range(P + 1)]

    if len(B_in) > 0 and i_skip >= 0:
        B = B_in.copy()

    i_max = n
    if sorted_two_piece:
        i_max = i_skip + 2

    B[0] = int(0)
    # Store minimum weight for each attainable profit state.
    for k in range(i_skip + 1, i_max):
        if p[k] == 0:
            continue
        A = B.copy()
        items_tmp = items.copy()

        for profit in range(p[k], P + 1):
            if A[profit - p[k]] < sys.maxsize and A[profit - p[k]] + w[k] < A[profit]:
                B[profit] = A[profit - p[k]] + w[k]
                if return_items:
                    items[profit] = items_tmp[profit-p[k]] + [k]
    #for p in range(P+1):
    #    if B[p] <= W:
    #        max_p = p

    return B, items #max_p


def for_loop_method_profit(
    P: int, p: np.ndarray, w: np.ndarray, W: int
) -> np.ndarray:
    """Return the legacy profit-state DP table up to capacity ``W``."""
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


def knapsack(w: np.ndarray, p: np.ndarray, W: int) -> float:
    """Return the best binary-knapsack value at capacity ``W``.

    This compatibility wrapper is retained for historical benchmark callers;
    new code should call one of the explicitly weight- or profit-indexed DP
    routines above.
    """
    B, _ = for_loop_method_all_w(np.array(p), np.array(w), int(W), np.array([], dtype=float), int(-1), False)
    return B[int(W)]




if __name__ == "__main__":
    
    """
    Code below used to track run times for the for loop method vs the scip method
    and can also be used to track run times for the f method vs g method (profit vs weight)
    """
    
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
        P_max = P_upper_bound(w_random, p_random, W_random)
        bounds.append(P_max)


        knap_start_process = time.process_time()
        knap_start_elapsed = time.time()
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

    print("The average processing time for knap and the for loop method respectively are: ", sum(knap_time_process)/float(len(fin_knap)), sum(for_time_process)/float(len(fin_for)))
    print("The average elapsed time for knap and the for loop method respectively are: ", sum(knap_time_elapsed)/float(len(fin_knap)), sum(for_time_elapsed)/float(len(fin_for)))
    # print("The average processing time for f, g, and knap respectively are: ",sum(f_time_process)/float(len(fin_f)), sum(g_time_process)/float(len(fin_g)), sum(knap_time_process)/float(len(fin_knap)))
    # print("The average elapsed time for f, g, and knap respectively are: ",sum(f_time_elapsed)/float(len(fin_f)), sum(g_time_elapsed)/float(len(fin_g)), sum(knap_time_elapsed)/float(len(fin_knap)))
    # print(for_loop_method_profit(P,p,w,W))
