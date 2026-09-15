#from pyscipopt import Model, quicksum, SCIP_PARAMSETTING
#from knapsack import rebppinit, update_rebpp
from rebpp import rebppinit_pyomo, print_sol, solve_instance
import numpy as np
import pandas as pd
import math
import time
import statistics as stat
import pyomo.environ as pe
from pyomo.opt import SolverStatus, TerminationCondition

__DEBUG = False
#if __name__ == "__main__":
NUM_ITEMS = 90 #20 #90
num_items = [20] #[20,30,60,90] #[30,60,90] #[60,90]
TIME_LIMIT = 7200
#3600
devProp = 0.4
c_const = 1.5 #2 #1.5

num_tests = 10
# Five robustness levels are considered, either 0%, 5%, 10%, 15% or 20% of sum(a_hat)
rob_level_mult = 0.1 #0 #0.05
# Four deadlines are generated for each instance, which are equal to a fraction of the sum of the worst-case job processing times; the fractions considered are 1/4, 1/6,1/8 and 1/10
V_mult = 1/8
#NZ_TOl = 1e-7
#GAPVAL1 = 0.4
#GAPVAL2 = 5e-2



def read_instance(i,sz,ss):
    fileName = "../data/" + str(sz) + "/" + str(sz) + "_(1,100)_"+ str(i) +".txt"
    #fileName = "../data/" + str(sz) + "/" + str(sz) + "_(1,20)_"+ str(i) +".txt"
    print("opening: ", fileName)
    test_data = pd.read_csv(fileName,skiprows=[1])
    #test_data = pd.read_csv("../data/" + str(sz) + "/" + str(sz) + "_(1,20)_"+ str(i) +".txt",skiprows=[1])
    a_bar = np.sort(test_data.iloc[range(ss),0])
    return a_bar

#test_data = pd.read_csv("../data/ma30.csv")
for sz in num_items: # [30, 60,90]:
    runTimes = []
    masterTimes = []
    runIter = []
    runBins = []
    runTimesWoTL = []
    masterTimesWoTL = []
    for instNum in range(num_tests):
        #test_data = pd.read_csv("../data/ma30.csv")
        #a_bar = test_data["a_bar_" + str(instNum)]
        # The processing-time deviation is 0.2 times the processing time, rounded to the nearest higher integer;
        a_bar = []
        if sz < 30:
            a_bar = read_instance(instNum, 30, min(NUM_ITEMS, sz))
        else:
            a_bar = read_instance(instNum,sz,min(NUM_ITEMS,sz))

        a_hat = (np.ceil(devProp*a_bar)).astype(int)
        a_bar = np.asarray(a_bar, dtype='int')
        Omega = int(math.ceil(rob_level_mult*sum(a_hat)))
        V = int(V_mult*(sum(a_hat)+sum(a_bar)))
        c_mult = c_const / V  # 3/(2*V) #2 / V  # 0.05

        n = len(a_bar)
        m = int(math.ceil(2*(sum(a_bar)+Omega)/V))
        c = np.ones(m)*c_mult
        print("Read file with ", n, " items", " m=", m)
        tmp, timeL, runTime, masterTime, numBins, scenario_num, theta_val, obj, cuts_added = solve_instance(a_bar, a_hat, V, c,Omega,TIME_LIMIT)

        if timeL == False and runTime < TIME_LIMIT-1e-6:
            runTimesWoTL.append(runTime)
            masterTimesWoTL.append(masterTime)
        runIter.append(scenario_num)
        runTimes.append(runTime)
        masterTimes.append(masterTime)
        runBins.append(numBins)

    print(" & {t1:.1f} & {t2:.1f} & {t3:.1f} & {t4:.1f} & {t5:.1f} & {t6:} & {t7:.1f} & {t8:} & {t9:} ".format(t1=stat.mean(runTimes),t2=max(runTimes),t3=stat.mean(masterTimes), t4=max(masterTimes), t5=stat.mean(runIter), t6=max(runIter),t7=stat.mean(runBins), t8=max(runBins), t9=num_tests - len(runTimesWoTL)))
    print(" & ", stat.mean(runTimesWoTL), " & ", max(runTimesWoTL), " & ", stat.mean(masterTimesWoTL), " & ", max(masterTimesWoTL))

    print("cuts_added: ",cuts_added)
#print(" & ", stat.mean(runTimes), " & ", max(runTimes), " & ", stat.mean(runIter), " & ", max(runIter), " & ", stat.mean(runBins), " & ", max(runBins))

#runTimesWoTL = list(filter(lambda x: x<TIME_LIMIT,runTimes))
#masterTimesWoTL = list(filter(lambda x: x<TIME_LIMIT, masterTimes))

      #sum(x >= TIME_LIMIT-0.2 for x in runTimes))

        # model.writeLP("after_update_model.lp")
        # def update_rebpp(model, a_bar, V, c, a, theta, y, f_bar,z):
