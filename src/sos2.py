from pyscipopt import Model, quicksum
import numpy as np
from numba import jit
import time
import random
# random.seed(0)


from knapsack import for_loop_method_all_w


B = 10
p = np.array([[0,0,6],
     [0,0,4],
     [0,0,5]])
b = np.array([[0,2,4],
     [0,2,3],
     [0,2,4]])

__DEBUG = True
__DEBUG_2 = False

def sos2(p,b,B):
    print("sos2....")
    model = Model("CPKP")
    model.setParam('limits/time', 3600)
    model.hideOutput()
    t = {}

    n,m = p.shape  # n = rows // m = columns
    nb,mb = b.shape

    assert m == mb and n == nb # make sure that they have the same dimensions

    for i in range(n):
        for j in range(m):
            t[i,j] = model.addVar(vtype="C", name="t(%s,%s)") # add variables to each position in t

    for i in range(n):
        model.addConsSOS2([t[i,j] for j in range(m)])
        model.addCons(quicksum(t[i,j] for j in range(m)) == 1) # check satisifies SOS2 properties

    model.addCons(quicksum(t[i,j] * b[i,j] for i in range(n) for j in range(m)) <= B, "width") # constraint to make sure product's less than B
    model.setObjective(quicksum(t[i,j]*p[i,j] for i in range(n) for j in range(m)), "maximize") # objective function
    model.optimize()

    fin = []
    i_max = None
    for i in range(n):
        for j in range(m):
            x = model.getVal(t[i,j])
            #print("x: ", i,j)
            # print(x)
            if x != 0:
                if j > 0 and j < m - 1:
                    i_max = i
                fin.append((i,j))
    if model.getStatus() != "optimal":
        #print("Optimal value: ", model.getObjVal())
        #print("objects: ", fin)
    #else:
        raise ValueError
    #print("B: ",B)
    #print("b: ", b)
    #print("p: ", p)
    print("sos i_max=",i_max)
    return model.getObjVal(), model, t, i_max

# p_eval - evaluate piecewise function: return profit value for item k, for a given x coordinate w
def p_eval(b,p,w,k):
    #print("p_eval....")
    b_row = b[k,:]
    p_row = p[k,:]

    b_max = b_row[-1]
    # print("b_max: ", b_max)
    # print("w: ", w)
    assert w <= b_max

    right_index = np.searchsorted(b_row,w)
    
    if w < 0:
        return -1*float('inf')
    
    if b_row[right_index] == w:
        return p_row[right_index]
    
    else:
        left_index = right_index - 1
        # print("left_index,right_index: ",left_index,right_index)
        difference = w - b_row[left_index]
        # print("difference: ",difference)
        fraction = difference / (b_row[right_index] - b_row[left_index])
        # print("fraction: ", fraction)
        # if w == 42:
            # print("hello: ",(1-fraction) * p_row[left_index] + (fraction) * p_row[right_index])
        return (1-fraction) * p_row[left_index] + fraction * p_row[right_index]
    
#need to pull out indices of items used in final knapsack in addition with imax
def convex_pw_knapsack_dp(p,b,W):
    print("convex_pw_knapsack_dp...")
    n,m = p.shape  # n = rows // m = columns
    if __DEBUG_2:
        for i in range(n):
            assert all(p[i,j+1] <= p[i,j+1] for j in range(len(p[i,:]) - 1))
            assert all(b[i,j] <= b[i,j+1] for j in range(len(b[i,:]) - 1))

    nb,mb = b.shape
    w_max = 0
    i_max = None
    items_max = [[]]
    assert m == mb and n == nb # make sure that they have the same dimensions

    initialP = 0
    for i in range(n):
        zeroIdxs = np.where(b[i,:]==0)[0]
        lastZero = max(zeroIdxs)
        initialP += p[i,lastZero]
        p[i,lastZero:m-1]=p[i,lastZero:m-1]-p[i,lastZero]

    profit_array = p[:,m-1]
    b_array = b[:,m-1]
    # print("b_array: ", b_array)
    
    max_val = 0
    for i in range(n):
        p_copy = profit_array.copy()
        b_copy = b_array.copy()
        temp_p = np.delete(p_copy,i)
        temp_b = np.delete(b_copy,i)
        B, items = for_loop_method_all_w(temp_p,temp_b,W)
        Wmax = W
        Wmin = max(W - b_array[i],0)
        if min(temp_b) > W:
            Wmin = 0
            Wmax = 1
        for w in range(Wmin, Wmax):   # after exluding item [i] loop on weight values between W-u[i] to W as new capacity
            merged_val = B[w] + p_eval(b,p,W-w,i)
            #if __DEBUG:
            #    if i == 7:
            #        print("i=", i, "merged_val=",merged_val, " w=", w," B[w]=", B[w], " p_eval=",p_eval(b,p,W-w,i))
            #        if w == W-1:
            #            print(B)
            #            print(temp_p,temp_b)
            if merged_val > max_val:
                max_val = merged_val
                w_max = w
                i_max = i
                items_max = items
                print("i_max: ", i_max)
                print("max_val=", max_val)
                #print("items max: ", items_max)
    #print("w_max: ",w_max)
    print("i_max: ",i_max)
    #print(max_val)
    #print(items_max)
    return initialP+max_val, items_max[w_max], i_max

#print(convex_pw_knapsack_dp(p,b,B))

# sos2(p,b,B)
# print(p_eval(b, p, 2.5, 1))
# print(b[1,:])
# print(np.searchsorted(b[1],2.5))

if __name__ == "__main__":
    sos2_time_process = []
    sos2_time_elapsed = []
    convex_time_process = []
    convex_time_elapsed = []
    fin_sos2 = []
    fin_convex = []
    k_random = 50

    for i in range(10):
        b = np.empty((0,3), int)
        p = np.empty((0,3), int)
        R = 100
        b_random = []
        
        p_random = np.random.rand(k_random)
        p_random *= R
        p_random = np.array(np.rint(p_random), dtype='i')
        
        for j in range(k_random):
            adjusted_weight_j = p_random[j] + int(R/10)
            b_random.append(adjusted_weight_j)
        
        for p_val in p_random:
            p = np.append(p, np.array([[0,0,p_val]]), axis=0)
            
        for b_val in b_random:
            b = np.append(b, np.array([[0, random.randint(0,b_val), b_val]]), axis = 0)

        B = int(20/101 * sum(b_random))
        
        sos2_start_process = time.process_time()
        sos2_start_elapsed = time.time()
        (sos2_val,_,_,_) = sos2(p,b,B)
        print("sos2 objVal=", sos2_val)
        sos2_end_process = time.process_time()
        sos2_end_elapsed = time.time()
        sos2_time_process.append(sos2_end_process - sos2_start_process)
        sos2_time_elapsed.append(sos2_end_elapsed - sos2_start_elapsed)

        convex_start_process = time.process_time()
        convex_start_elapsed = time.time()
        (convex_val,_,_) = convex_pw_knapsack_dp(p,b,B)
        print("convex_pw_knapsack_dp objVal=", convex_val)
        convex_end_process = time.process_time()
        convex_end_elapsed = time.time()
        convex_time_process.append(convex_end_process - convex_start_process)
        convex_time_elapsed.append(convex_end_elapsed - convex_start_elapsed)

        assert abs(sos2_val-convex_val) < 1e-8

        fin_sos2.append(sos2_val)
        fin_convex.append(convex_val)
        # print("p: ",p)
        # print("b: ",b)
        # print("fin_sos2: ", fin_sos2)
        # print("fin_convex: ", fin_convex)

    print("The average processing time for sos2 and the for convex method respectively are: ", sum(sos2_time_process)/float(len(fin_sos2)), sum(convex_time_process)/float(len(fin_convex)))
    print("The average elapsed time for sos2 and the for convex method respectively are: ", sum(sos2_time_elapsed)/float(len(fin_sos2)), sum(convex_time_elapsed)/float(len(fin_convex)))
        # print(p_eval(b,p,6,2))