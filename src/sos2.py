from json.encoder import INFINITY

from pyomo.common.config import NonNegativeFloat
from pyscipopt import Model, quicksum
import numpy as np
from numba import jit
from numba.typed import List
import time
import random
import math
# random.seed(0)
from gurobipy import GRB 
import pyomo.environ as pe
import pandas as pd

from knapsack import for_loop_method_all_w, generate_random_instance, for_loop_method_all_w_save_all


#W = 10
#p = np.array([[0,0,6],
#     [0,0,4],
#     [0,0,5]])
#b = np.array([[0,2,4],
#     [0,2,3],
#     [0,2,4]])

#n = len(p)

__DEBUG = False
__DEBUG_2 = False
__DEBUG_3 = False

NZ_TOL = 1e-10
TIMELIMIT = 1800

def sos2(p,b,B):
    if __DEBUG_2:
        print("sos2....")
    model = Model("CPKP")
    model.setParam('limits/time', TIMELIMIT)
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

    sos2_start_process = time.process_time()
    #sos2_start_elapsed = time.time()
    model.optimize()
    sos2_end_process = time.process_time()
    #sos2_end_elapsed = time.time()
    sos2_time_process = sos2_end_process - sos2_start_process
    #sos2_time_elapsed = sos2_end_elapsed - sos2_start_elapsed
    solve_time = model.getSolvingTime()
    fin = []
    i_max = None
    for i in range(n):
        for j in range(1,m):
            if p[i,m-1]>0 and model.getVal(t[i,j]) > NZ_TOL:
                if j < m - 1:
                    i_max = i
                    break
                elif j == m-1:
                    fin.append(i)
    status = model.getStatus()
    if status != "optimal": # and status!="timelimit":
        #raise Exception("SoS solution is not optimal: " + status)
        print("ERROR - suboptimal SoS solution")
    if __DEBUG_2:
        print(" objVal=", model.getObjVal(), " sos i_max=", i_max)
    return model.getObjVal(), fin, i_max, solve_time, sos2_time_process


def sos2_gurobi(p,b,B):
    n,m = p.shape  # n = rows // m = columns
    nb,mb = b.shape

    md = pe.ConcreteModel()
    md.I = pe.RangeSet(0,n-1)
    md.J = pe.RangeSet(0,m-1)

    md.t = pe.Var(md.I,md.J,domain=pe.NonNegativeReals)
    md.obj = pe.Objective(expr = sum(p[i,j]*md.t[i,j] for i in md.I for j in md.J),sense=pe.maximize)
    #def c_rule(md):
    #    return
    md.c = pe.Constraint(expr=sum(b[i, j]*md.t[i, j] for i in md.I for j in md.J) <= B)

    def rule_sossum(md,i):
        return sum(md.t[i, j] for j in md.J)==1
    md.sossum = pe.Constraint(md.I, rule=rule_sossum)

    def rule_mysos(md,i):
        return [md.t[i,j] for j in md.J]
    md.mysos = pe.SOSConstraint(md.I, rule=rule_mysos, sos=2)
    opt = pe.SolverFactory('gurobi_direct')
    opt.options['TimeLimit'] = TIMELIMIT
    #opt.options['threads'] = 1
    grb_start_process = time.process_time()
    grb_start_elapsed = time.time()
    results = opt.solve(md,tee=False)
    grb_start_process_end = time.process_time()
    grb_start_elapsed_end = time.time()
    objVal = md.obj()
    fin = []
    i_max = None
    if 'ok' == str(results.Solver.status):
        for i in md.I:
            if p[i, m - 1] > 0:
                for j in md.J:
                    if md.t[i, j]() > NZ_TOL:
                        if md.t[i, j]() < 1-NZ_TOL:
                            i_max = i
                            break
                        elif j == m - 1:
                            fin.append(i)
    else:
        print("No Valid Solution Found")
        objVal = -INFINITY
    return objVal, fin, i_max, grb_start_elapsed_end - grb_start_elapsed, grb_start_process_end - grb_start_process


#N = 1
#model = pyo.ConcreteModel()
#model.x = pyo.Var([1], domain=pyo.NonNegativeReals, bounds=(0,40))
#model.A = pyo.Set(initialize=[1,2,4,6])
#model.y = pyo.Var(model.A, domain=pyo.NonNegativeReals, bounds=(0,2))
#model.OBJ = pyo.Objective(
#    expr=(1*model.x[1]+
#          2*model.y[1]+
#          3*model.y[2]+
#          -0.1*model.y[4]+
#          0.5*model.y[6])
#    )
#model.ConstraintYmin = pyo.Constraint(
#    expr = (model.x[1]+
#            model.y[1]+
#            model.y[2]+
#            model.y[6] >= 0.25
#            )
#    )
#model.mysos = pyo.SOSConstraint(
#    var=model.y,
#    sos=N
#    )
#solver_name = 'scip'
#solver_available = bool(check_available_solvers(solver_name))
#if solver_available:
#     opt = pyo.SolverFactory(solver_name)
#     opt.solve(model, tee=False)
#     assert isclose(pyo.value(model.OBJ), 0.05, abs_tol=1e-3)



# p_eval - evaluate piecewise function: return profit value for item k, for a given x coordinate w
@jit(nopython=True)
def p_eval(b_row, p_row, w):
    #print("p_eval....")
#    b_row = b[k,:]
#    p_row = p[k,:]
    b_max = b_row[-1]
    # print("b_max: ", b_max)
    # print("w: ", w)
    assert 0 <= w <= b_max
    if w==0:
        return p_row[0]
    right_index = np.searchsorted(b_row,w)
    if b_row[right_index] == w:
        return p_row[right_index]
    else:
        left_index = right_index - 1
        # print("left_index,right_index: ",left_index,right_index)
        assert b_row[right_index] > b_row[left_index]
        difference = float(w - b_row[left_index])
        assert difference >= 0
        # print("difference: ",difference)
        fraction = difference / float(b_row[right_index] - b_row[left_index])
        # print("fraction: ", fraction)
        return (1-fraction) * p_row[left_index] + fraction * p_row[right_index]

@jit(cache=True,nopython=True)
def sort_instance_by_slopes(p, b):
    dp_db = (p[:,2]-p[:,1])/(b[:,2]-b[:,1])
    indexes = np.argsort(dp_db)
    indexes = np.flip(indexes) # need them in decreasing order
    return p[indexes,:], b[indexes,:]


#need to pull out indices of items used in final knapsack in addition with imax
@jit(nopython=True)
def convex_pw_knapsack_dp(p, b, W, y_intercept_nonzero=False):
    if __DEBUG_2:
        print("convex_pw_knapsack_dp...")
    n,m = p.shape # n = rows // m = columns
#    if __DEBUG_3:
#        for i in range(n):
#            if not all(p[i,j] <= p[i,j+1] for j in range(m-1)):
#                print(p,b)
#                raise OSError("nonconvex p")
#            if not all(b[i,j] <= b[i,j+1] for j in range(m-1)):
#                print(p,b)
#                raise OSError("nonconvex b")
    nb,mb = b.shape
    w_max = 0
    i_max = None
    #items_max = [[i for i in range(0)] for _ in range(W+1)]#[[]]
    assert m == mb and n == nb # make sure that they have the same dimensions
    initialP = 0
    if y_intercept_nonzero:
        for i in range(n):
            zeroIdxs = np.where(b[i,:]==0)[0]
            lastZero = max(zeroIdxs)
            initialP += p[i,lastZero]
            p[i,lastZero:m-1]=p[i,lastZero:m-1]-p[i,lastZero]

    profit_array = p[:,m-1]
    b_array = b[:,m-1]
    max_val = float(0)
    B_all = np.zeros((n,W+1),dtype=float)
    B = np.zeros(W+1)
    for i in range(n-1,-1,-1):  #
        pi = profit_array[i]
        profit_array[i] = 0
        skip_idx = i
        if i == 0 or i == n-1:
            skip_idx = -1
        if i == n-1:
            #tmpB = np.zeros(W + 1)
            #tmpL = [List().append(-1) for _ in range(1)]
            for_loop_method_all_w_save_all(profit_array, b_array, W, B_all)
            #np.copyto(B,B_all[n-1,:])#
            B = B_all[n-1,:].copy()
        else:
            #skip_idx=-1
            B, _ = for_loop_method_all_w(profit_array, b_array, W, B_all[max(skip_idx-1,-1),:], skip_idx, False,True) #items_all[i-1], skip_idx)
        profit_array[i] = pi  # allow i to be selected again
        w_max = W #max(W - b_array[i],1) #W
        w_min = max(W - b_array[i],1)
        if min(np.delete(b_array,i)) > W:
            #w_min = 0
            w_max = 1
        for w in range(w_min, w_max):   # after exluding item [i] loop on weight values between W-u[i] to W as new capacity
            merged_val = B[w] + p_eval(b[i,:],p[i,:],W-w)

            if merged_val > max_val:
                max_val = merged_val
                #print("i=", i, " merged_val=", merged_val, " peval=", p_eval(b[i,:],p[i,:],W-w))
                w_max = w
                i_max = i
    if __DEBUG_2:
        print("convex_pw_knapsack_dp i_max=", i_max)
    p_vec = profit_array.copy()
    b_vec = b_array.copy()
    if i_max is not None:
        p_vec[int(i_max)] = 0
        b_vec[int(i_max)] = W
    _, items_max = for_loop_method_all_w(p_vec, b_vec, W) #, B, -1, True)  # items_all[i-1], skip_idx)
    #B, items_max = for_loop_method_all_w(profit_array, b_array, W, B_all[max(i_max-1,-1),:], i_max, True)  # items_all[i-1], skip_idx)
    #print("i_max: ",i_max)
    #print(max_val)
    #print(items_max)
    if __DEBUG_2:
        print(items_max[w_max])
        print("convex_pw_knapsack_dp objVal=", initialP+max_val, " i_max=", i_max, " w_max=", w_max, " items_max[w_max]=", items_max[w_max])
    return initialP+max_val, items_max[w_max], i_max

#print(convex_pw_knapsack_dp(p,b,B))

# sos2(p,b,B)
# print(p_eval(b, p, 2.5, 1))
# print(b[1,:])
# print(np.searchsorted(b[1],2.5))

def read_instance(i):
    test_data = pd.read_csv("../data/Original_Instances/InverseStrong00"+ str(i) ,skiprows=3, sep="\\s+")
    return test_data.iloc[:,0], test_data.iloc[:,1]

def write_instance(p_random,b_random,i):
    df = pd.DataFrame(data=np.column_stack((b_random,p_random)))
    fileName = "data" + str(len(b_random)) + "_" + str(len(p_random)) + "_" + str(i) + ".txt"
    #f = open(fileName,"w")
    #f.write("\n\n", length(p_random), "\n")
    df.to_csv(fileName)

if __name__ == "__main__":
    random.seed(101)
    for R in [100]: #[1000,10000]:
        for k_random in [50,100,250,500]: #, 250]: #[50, 100, 150, 200, 250, 300]:
            sos2_time_process = []
            sos2_time_elapsed = []
            sos2g_time_process = []
            sos2g_time_elapsed = []
            convex_time_process = []
            convex_time_elapsed = []
            fin_sos2 = []
            fin_sos2g = []
            fin_convex = []

            for i in range(1,30):
    #for i in range(1, 9):

                b = np.empty((0,3), int)
                p = np.empty((0,3),float)
                p_random,b_random = generate_random_instance(k_random,R,True)
                write_instance(p_random,b_random,i)
                #b_random, p_random  = read_instance(i)
                k_random = len(b_random)

                for p_val in p_random:
                    p = np.append(p, np.array([[0, 0, p_val+1e-3*random.random()]]), axis=0)

                for b_val in b_random:
                    #b = np.append(b, np.array([[0, random.randint(0,b_val), b_val]]), axis = 0)
                    b = np.append(b, np.array([[0, b_val-1, b_val]]), axis = 0)

                        #print(for_loop_method_all_w(p, b, 63,-1,[i for i in range(1)], [[i for i in range(1)] for _ in range(1)], -1))
                W = int(math.ceil((5+i*3)/101 * sum(b_random)))

                p, b = sort_instance_by_slopes(p, b)  # sort instance for linear speed up

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
                sos2_val = sos2g_val
        ##################################

                convex_start_process = time.process_time()
                convex_start_elapsed = time.time()
                convex_val,items,imax = convex_pw_knapsack_dp(p,b,W)
                convex_end_process = time.process_time()
                convex_end_elapsed = time.time()
                convex_time_process.append(convex_end_process - convex_start_process)
                convex_time_elapsed.append(convex_end_elapsed - convex_start_elapsed)

                if abs(sos2g_val-convex_val)/sos2g_val > 1e-4 and int(sos2_time_elapsed[-1]) < 3599:
                    print("ERR ! ................  sos2g objval: ", sos2g_val, " DP val: ", convex_val)
                    #print("W=", W, " sosg_imax=", sosg_imax, " imax=", imax, "items=", items, " sum(b other than imax)=", quicksum(b[items,-1]), " sum(p other than imax)=",quicksum(p[items,-1]), " t=", sos_items)
                    raise Exception("different obj vals")


                fin_sos2.append(sos2_val)
                fin_sos2g.append(sos2g_val)
                fin_convex.append(convex_val)

          #  print(k_random, " & ", f"{sum(sos2_time_process)/float(len(fin_sos2)):.2f}", " & ",  f"{max(sos2_time_process):.2f}" , " & ",  f"{sum(convex_time_process)/float(len(fin_convex)):.2f}", " & ", f"{max(convex_time_process):.2f}")
          #  print(k_random, " & ", f"{sum(sos2_time_elapsed)/float(len(fin_sos2)):.2f}", " & ", f"{max(sos2_time_elapsed):.2f}" , " & ", len([x for x in sos2_time_elapsed if x>TIMELIMIT-1]), " & ",  f"{sum(convex_time_elapsed)/float(len(fin_convex)):.2f}", " & ", f"{max(convex_time_elapsed):.2f}", " & ", len([x for x in convex_time_elapsed if x>TIMELIMIT-1]) )

            print(k_random, " & ", f"{sum(sos2g_time_process) / float(len(fin_sos2g)):.2f}", " & ", f"{max(sos2g_time_process):.2f}", " & ", f"{sum(convex_time_process) / float(len(fin_convex)):.2f}", " & ", f"{max(convex_time_process):.2f}")
            print(k_random, " & ", f"{sum(sos2g_time_elapsed) / float(len(fin_sos2g)):.2f}", " & ", f"{max(sos2g_time_elapsed):.2f}", " & ", len([x for x in sos2g_time_elapsed if x > TIMELIMIT-1]), " & ",f"{sum(convex_time_elapsed) / float(len(fin_convex)):.2f}", " & ", f"{max(convex_time_elapsed):.2f}", " & ", len([x for x in convex_time_elapsed if x > TIMELIMIT-1]))
    # print(p_eval(b,p,6,2))
