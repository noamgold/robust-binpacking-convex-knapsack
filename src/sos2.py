from pyscipopt import Model, quicksum
import numpy as np
from numba import jit

from knapsack import for_loop_method_all_w



B = 10
p = np.array([[0,0,6],
     [0,0,4],
     [0,0,5]])
b = np.array([[0,2,4],
     [0,2,3],
     [0,2,4]])

def sos2(p,b,B):

    model = Model("CPKP")

    t = {}

    n,m = p.shape  # n = rows // m = columns
    nb,mb = b.shape


    assert m == mb and n == nb # make sure that they have the same dimensions

    for i in range(n):
        for j in range(m):
            t[i,j] = model.addVar(vtype="C", name="x(%s,%s)") # add variables to each position in t

    for i in range(n):
        model.addConsSOS2([t[i,j] for j in range(m)])
        model.addCons(quicksum(t[i,j] for j in range(m)) == 1) # check satisifies SOS2 properties

    model.addCons(quicksum(t[i,j] * b[i,j] for i in range(n) for j in range(m)) <= B, "width") # constraint to make sure product's less than B
    model.setObjective(quicksum(t[i,j]*p[i,j] for i in range(n) for j in range(m)), "maximize") # objective function
    model.optimize()

    fin = []
    for i in range(n):
        for j in range(m):
            x = model.getVal(t[i,j])
            print(x)
            if x != 0:
                fin.append((i,j))

    if model.getStatus() == "optimal":
        print("Optimal value: ", model.getObjVal())
        print("objects: ", fin)

    return model.getObjVal()

# sos2(p,b,B)

def p_eval(b,p,w,k):
    b_row = b[k,:]
    p_row = p[k,:]

    b_max = b_row[-1]
    print("b_max: ", b_max)
    print("w: ", w)
    assert w <= b_max

    right_index = np.searchsorted(b_row,w)
    
    if b_row[right_index] == w:
        return p_row[right_index]
    else:
        left_index = right_index - 1
        difference = w - b_row[left_index]
        fraction = difference / (b_row[right_index] - b_row[left_index])
        return fraction * p_row[left_index] + (1-fraction) * p_row[right_index]
    
def convex_pw_knapsack(b,p,W):

    n,m = p.shape  # n = rows // m = columns
    nb,mb = b.shape

    assert m == mb and n == nb # make sure that they have the same dimensions

    profit_array = p[:,m-1]
    b_array = b[:,m-1]
    print("b_array: ", b_array)
    
    max_val = 0
    for i in range(n):
        p_copy = profit_array.copy()
        b_copy = b_array.copy()
        temp_p = np.delete(p_copy,i)
        temp_b = np.delete(b_copy,i)
        B = for_loop_method_all_w(temp_p,temp_b,W)
        for j in range(W - b_array[i], W):
            merged_val = B[j] + p_eval(b,p,W-j,i)
            if merged_val > max_val:
                max_val = merged_val
    return max_val


print(convex_pw_knapsack(b,p,B))

# sos2(p,b,B)
# print(p_eval(b, p, 2.5, 1))
# print(b[1,:])
# print(np.searchsorted(b[1],2.5))

k = np.array([[0,0,6],
     [0,0,4],
     [0,0,5],
     [1,3,4]])
print(k.shape)