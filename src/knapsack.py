from pyscipopt import Model, quicksum
from numba import jit
from numba.typed import List

import random
import time
import numpy as np
import math
np.random.seed(0)
W = 8
w = [1,2,6, 2]
p = [2,3,6, 3]
#k = 3
P = sum(p)

def RandomBinPacking(n, B):
    fin = []
    for _ in range(n):
        fin.append(random.random()*B)
    return fin, B

def BinPackingExample():
    B = 9
    w = [2,3,4,5,6,7,8]
    q = [4,2,6,6,2,2,2]
    s=[]
    for j in range(len(w)):
        for i in range(q[j]):
            s.append(w[j])
    return s,B

def FFD(s, B):
    remain = [B]
    sol = [[]]
    for item in sorted(s, reverse=True):
        for j,free in enumerate(remain):
            if free >= item:
                remain[j] -= item
                sol[j].append(item)
                break
        else:
            sol.append([item])
            remain.append(B-item)
    return sol


def bpp(s,B):
    n = len(s)
    U = len(FFD(s,B))
    model = Model("bpp")
    x,y = {},{}
    for i in range(n):
        for j in range(U):
            x[i,j] = model.addVar(vtype="B", name="x(%s,%s)"%(i,j))
    for j in range(U):
        y[j] = model.addVar(vtype="B", name="y(%s)"%j)
    for i in range(n):
        model.addCons(quicksum(x[i,j] for j in range(U)) == 1, "Assign(%s)"%i)
    for j in range(U):
        model.addCons(quicksum(s[i]*x[i,j] for i in range(n)) <= B*y[j], "Capac(%s)"%j)
    for j in range(U):
        for i in range(n):
            model.addCons(x[i,j] <= y[j], "Strong(%s,%s)"%(i,j))
    model.setObjective(quicksum(y[j] for j in range(U)), "minimize")
    model.data = x,y
    return model

def solveBinPacking(s,B):
    n = len(s)
    U = len(FFD(s,B))
    model = bpp(s,B)
    x,y = model.data
    print(U+1)
    model.setObjlimit(U+1)
    model.optimize()
    bins = [[] for i in range(int(model.getObjVal()))]

    for (i,j) in x:
        if model.getVal(x[i,j]) > .5:
            bins[j].append(s[i])

    for i in range(bins.count([])):
        bins.remove([])

    for b in bins:
        b.sort()

    bins.sort()

    return bins

def min_weight(P,k,w,p):
    if P == 0:
        return 0
    if k == 0:
        if P == p[0]:
            return w[0]
        else:
            return float('inf')
    else:
        new_k = k-1
        if p[k-1] > P and k > 0:
            return min_weight(P,new_k, w,p)
        return min(min_weight(P,new_k,w,p), min_weight(P-p[k-1], new_k, w,p) + w[k-1])

def p_star(p_bar,w,p):
    print(p_bar)
    fin = []
    for i in range(p_bar):
        print(i)
        if min_weight(i,len(w),w,p) <= W:
            fin.append(i)
    print(fin)
    return fin[-1]

def f(P,k,w,p):
    # print(P,k)
    if P <= 0:
        return 0
    if k == 1:
        if P <= p[0]:
            return w[0]
        else:   # P>p[0]
            return float('inf')
    else:
        new_k = k-1
        # if k>=2:
        #     return min_weight(P,new_k, w,p)
        if k >= 2:
            return min(f(P,new_k,w,p), f(P-p[k-1], new_k, w,p) + w[k-1])
        
def p_opt_trial(P_bar,k,w,p,W):
    fin = []
    # print("hello")
    # print("P_Bar val: ", P_bar)
    for P in range(P_bar+1):
        print(P,":", f(P,k,w,p))
        if f(P,k,w,p) <= W:
            fin.append(P)
    # print("P=2835: ", f(2835,k,w,p))
    # print(W)
    return fin[-1]
    
def p_opt(P,k,w,p,W):
    l = 1
    r = P

    while l < r:
        x = (r-l+1)/2
        # print(x)
        mid = math.floor(l + x)
        print("mid: ", mid)
        print("f: ", f(mid,k,w,p))
        print("W: ", W)
        if f(mid,k,w,p) == W:
            return mid
        
        if f(mid,k,w,p) <= W:
            # print("<")
            l = mid
        
        elif f(mid,k,w,p) > W:
            # print(">")
            r = mid - 1
    #     print("l: ",l)
    #     print("r: ",r)
    # print("P=1318: ", f(1318,k,w,p))
    return l

def g(p,w,W,k):
    if k == 0 and W < w[0] and 0 <= W:
        return 0
    if W < 0:
        return -1 * float('inf')
    if W >= w[0] and k == 1:
        return p[0]
    else:
        new_k = k-1
        if k>=1 and W<w[k-1]:
            return g(p,w,W,new_k)
        return max(g(p,w,W,new_k), g(p, w, W-w[k-1],new_k)+p[k-1])
    
def knapsack(w,p,W):

    m = len(w)
    knapsack = Model("KP")
    y = {}
    for i in range(m):
        y[i] = knapsack.addVar(vtype="B", name="y[%d]"%i)
    knapsack.addCons(quicksum(w[i]*y[i] for i in range(m)) <= W, "width")
    knapsack.setObjective(quicksum(p[i]*y[i] for i in range(m)), "maximize")
    knapsack.optimize()
    fin = []
    for i in range(m):
        x = knapsack.getVal(y[i])
        if x != 0:
            fin.append(i)

    # if knapsack.getStatus() == "optimal":
        # print("Optimal value: ", knapsack.getObjVal())
        # print("objects: ", fin)
    return knapsack.getObjVal()

def P_upper_bound_real(w,p,W):
    curr_total_weight = 0
    p_bar = 0
    w_array = np.array(w)
    p_array = np.array(p)
    ratios = np.divide(p_array,w_array)
    indexes = np.argsort(ratios)
    reversed_indexes = np.flip(indexes)
    for index in reversed_indexes:
        weight = w_array[index]
        profit = p_array[index]
        # if W - curr_total_weight > 0:
        if curr_total_weight + weight < W:
            curr_total_weight += weight
            p_bar += profit
        else:
            fraction = (W - curr_total_weight)/weight
            p_bar += profit * fraction
            return math.ceil(p_bar)

def P_upper_bound(w,p,W):
    """
    function to get a higher p_upper bound rather than just the sum
    """
    curr_total_weight = 0
    p_bar = 0
    w_to_p = {}
    ratios = {}
    for i in range(len(w)):
        w_to_p[w[i]] = p[i]
        ratios[(p[i]/w[i])] = w[i]
    sorted_bfb = dict(reversed(sorted(ratios.items()))) #orderes items in which they give the most bang for your buck
    for best in sorted_bfb:
        weight = sorted_bfb[best]
        if W - curr_total_weight > 0:
            if curr_total_weight + weight < W:
                curr_total_weight += weight
                p_bar += w_to_p[weight]
            else:
                fraction = (W - curr_total_weight)/weight
                p_bar += w_to_p[weight] * fraction
                return p_bar
        else:
            return p_bar

@jit(nopython=True)
def for_loop_method_all_w_save_all(p,w,W,B_all):
    n = len(p)
    #B_all = np.zeros((W+1,n),dtype=float)
    for k in range(0, n):
        if k >= 1:
            #np.copyto(B_all[k,:],B_all[k-1,:])
            B_all[k, :] = B_all[k-1,:].copy()
        if p[k] > 0:
            for weight in range(w[k], W + 1):
                if k == 0:
                    B_all[0,weight] = p[0]
                else:
                    if B_all[k-1,weight - w[k]] + p[k] > B_all[k-1,weight]:
                        B_all[k,weight] = B_all[k-1,weight - w[k]] + p[k]


# vector implementation of DP - cost version
# save B and items after item i in addition to final ones, start from i_skip + 1 and save at
@jit(nopython=True)
def for_loop_method_all_w(p,w,W, B_in = np.array([]), i_skip = int(-1), return_items = True, sorted_two_piece = False):
#def for_loop_method_all_w(p, w, W, B_in=np.array([],dtype=float), items_in=[List().append(-1) for _ in range(W+1)], i_skip=int(-1)):
    n = len(p)
    nn = len(w)
    assert W > 0
    if n != nn:
        raise Exception("n!=n")
    B = np.zeros(W+1) #[0] * (W + 1)
    items = [[i for i in range(0)] for _ in range(W+1)] #used to initialize a 2d array
    #items_all = np.full([W+1,n],[-1])  #[[[i for i in range(0)] for _ in range(W+1)] for _ in range(n)]

    if np.min(w) <= W:
        if len(B_in) > 0 and i_skip >= 0:
           B = B_in.copy()
           #print("copied B")
           #items = items_in.copy()
        else:
          i_skip = -1
        i_max = n
        if sorted_two_piece:
            i_max = i_skip+2
        for k in range(i_skip+1,i_max):
            if p[k] == 0:
                continue
            A = B.copy()
            items_tmp = items.copy()
            for weight in range(w[k], W + 1):
                if A[weight - w[k]] + p[k] > A[weight]:
                    B[weight] = A[weight - w[k]] + p[k]
                    if return_items:
                        items[weight] = items_tmp[weight-w[k]] + [k] #[*items[weight-w[k]], k]  # items[weight-w[k]] + [k]#[*items[weight-w[k]], k]
            #if save_all:
            #    B_all[:,i] = B.copy()
            #    items_all[i] = items.copy()
    #if not np.any(B):
    #    raise Exception("zero B at the end of knapsack DP")

    return B, items#, B_all, items_all

#@jit(nopython=True)
#def for_loop_method(p,w,W):
#    B, items = for_loop_method_all_w(p,w,W)
#    return B[W]

def generate_random_instance(k_random, R, inversely_cor=True):
    b_random = np.zeros(k_random, dtype=int)
    p_random = np.zeros(k_random, dtype=float)
    if inversely_cor:
        p_random = np.random.rand(k_random)*R
        p_random = np.ceil(p_random) #np.array(np.ceil(p_random), dtype='i')
        for j in range(k_random):
            b_random[j] = p_random[j] + int(R / 10)  # inverse strongly correlated
    else:  # almost strongly correlated   #mstr
        d = 6
        k1 = 3*R/10
        k2 = 2*R/10
        #b_random = 1+np.random.rand(k_random) * (R-1)
        #b_random = np.ceil(b_random).astype(int)
        #b_div_6 = b_random.astype(int)%d
        #b_div_6[b_div_6 > 0] = -1
        #b_div_6[b_div_6==0] = 1
        #b_div_6[b_div_6 == -1] = 0
        #p_random = b_random + k2
        #p_random[b_div_6] = b_random[b_div_6] +k1
        b_random = np.random.rand(k_random)*(R-1)
        b_random = 1+(np.ceil(b_random)).astype(int)
        for j in range(k_random):
            p_random[j] = random.randint(math.floor(b_random[j]+R/10-R/500),math.ceil(b_random[j]+R/10+R/500))
    return p_random,b_random


def for_loop_method_profit(P,p,w,W):
    n = len(p)
    B = [float('inf')] * (P + 1)
    B[0] = 0

    for k in range(n):
        A = B.copy()

        for profit in range(p[k], P + 1):
            if A[profit - p[k]] + w[k] < A[profit]:
                B[profit] = A[profit - p[k]] + w[k]

    for p in range(P+1):
        if B[p] <= W:
            max_p = p

    return max_p

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
        P_max = P_upper_bound_real(w_random, p_random, W_random)
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