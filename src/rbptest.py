from pyscipopt import Model, quicksum, SCIP_PARAMSETTING
#from knapsack import rebppinit, update_rebpp
from sos2 import sos2, convex_pw_knapsack_dp
from rebpp import rebppinit, update_rebpp, convex_pw_knapsack_wrapper, print_sol
import numpy as np
import csv
import pandas as pd
import math
import time
import statistics as stat


__DEBUG = False
#if __name__ == "__main__":

VIOL_TOL = 1e-3
INT_TOL = 1e-3
TIMELIMIT = 3600.0
devProp = 0.2
num_tests = 10
# Five robustness levels are considered, either 0%, 5%, 10%, 15% or 20% of sum(a_hat)
rob_level_mult = 0.1 #0 #0.05
# Four deadlines are generated for each instance, which are equal to a fraction of the sum of the worst-case job processing times; the fractions considered are 1/4, 1/6,1/8 and 1/10
V_mult = 1/8
NZ_TOl = 1e-7
GAPVAL1 = 0.2
GAPVAL2 = 5e-2


#VIOL_TOL = 1e-6
#devProp = 0.2
#num_tests = 10
# Five robustness levels are considered, either 0%, 5%, 10%, 15% or 20% of sum(a_hat)
#rob_level_mult = 0.1 #0 #0.05
# Four deadlines are generated for each instance, which are equal to a fraction of the sum of the worst-case job processing times; the fractions considered are 1/4, 1/6,1/8 and 1/10
#V_mult = 1/8
#NZ_TOl = 1e-7
#INT_TOL = 1e-3
#gapVal = 0.2

runTimes = []
runIter = []
runBins = []

def read_instance(i):
    test_data = pd.read_csv("../data/30/30_(1,20)_"+ str(i) +".txt",skiprows=[1])
    return test_data.iloc[:,0]

#test_data = pd.read_csv("../data/ma30.csv")

for instNum in range(num_tests):
    #test_data = pd.read_csv("../data/ma30.csv")
    #a_bar = test_data["a_bar_" + str(instNum)]
    # The processing-time deviation is 0.2 times the processing time, rounded to the nearest higher integer;
    a_bar = read_instance(instNum)
    a_hat = (np.ceil(devProp*a_bar)).astype(int)
    a_bar = np.asarray(a_bar, dtype='int')
    Omega = int(math.ceil(rob_level_mult*sum(a_hat)))
    V = int(V_mult*(sum(a_hat)+sum(a_bar)))
    c_mult = 2 / V  # 0.05

    n = len(a_bar)
    m = int(math.ceil(2*(sum(a_bar)+Omega)/V))
    c = np.ones(m)*c_mult
    print("Read file with ", n, " items", " m=", m)

    alpha = {}
    scenario_num = 0

    gapVal = GAPVAL1
    start = time.time()
    model, theta, y, f_bar, z = rebppinit(a_bar, a_hat, V, c)
    model.setIntParam("display/verblevel", 2)
    model.setIntParam("display/freq", 10000)
    model.setRealParam("limits/time", TIMELIMIT)
    #model.hideOutput

    it = 0
    numBins = 0
    while True:
        f = {}
        u = {}
        model.setRealParam('limits/gap', gapVal)
        model.optimize()
        # print_sol(model)
        # model.writeProblem("model" + str(iter) + ".cip",trans=False)
        # model = model2
        b = np.zeros((m, 3), int)
        p = np.zeros((m, 3), float)
        solStatus = model.getStatus()
        if solStatus != "optimal" and solStatus != "gaplimit":
            print("Error (suboptimal):", solStatus)
        #    raise ValueError
        theta_star = model.getVal(theta)
        numBins = 0
        for j in range(m):
            if model.getVal(y[j]) > 1-INT_TOL:
                f[j] = 0
                u[j] = 0
                numBins += 1
                for i in range(n):
                    # print("loop problem")
                    if model.getVal(z[i, j]) > 1-INT_TOL:
                        f[j] += a_bar[i]
                        u[j] += a_hat[i]
                b[j,1] = max(V - f[j], 0)
                b[j,2] = u[j] #,axis=0) #max(u[j] - V + f[j], 0)]]), axis=0)
                p[j,0] = c[j] * max(f[j] - V, 0)
                p[j,1] = p[j,0]
                p[j,2] = c[j] * max(u[j] - V + f[j], 0)
            #p = np.append(p, np.array([[c[j] * max(f[j] - V, 0), c[j] * max(f[j] - V, 0), c[j] * max(u[j] - V + f[j], 0)]]), axis=0)
        #print(b)
        #print(p)
        if __DEBUG:
            print("Before running convex knapsack, Omega=", Omega, " p=", p, " b=", b)
        #p_star = 0
        #if np.sum(p[:,2]) > 0: #NZ_TOl:
        p_star, a = convex_pw_knapsack_wrapper(p, b, Omega, z, a_hat, model, False)

        it += 1
        print("iteration: ", it, " p_star val: ", p_star, " theta_star_val: ", theta_star, " ******")

        # corrected 21/3
        if p_star <= theta_star + VIOL_TOL:
            if gapVal == GAPVAL2:
                print("terminating, could not find a constraint violating by more than tol=", VIOL_TOL, " p_star=", p_star)
                print_sol(model)
                break
            else:
                gapVal = GAPVAL2
                print("setting gapVal: ", gapVal)

        model, alpha = update_rebpp(model, a_bar, V, c, a, theta, y, z, alpha, scenario_num)
        scenario_num += 1
    runTime = time.time()-start
    print("Elapsed time instance instNum=", instNum, " elapsed time: ", runTime)
    runIter.append(scenario_num)
    runTimes.append(runTime)
    runBins.append(numBins)

print(" & ", stat.mean(runTimes), " & ", max(runTimes), " & ", stat.mean(runIter), " & ", max(runIter), " & ", stat.mean(runBins), " & ", max(runBins))

        # model.writeLP("after_update_model.lp")
        # def update_rebpp(model, a_bar, V, c, a, theta, y, f_bar,z):