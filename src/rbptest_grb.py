"""Runtime benchmark for the REBP algorithm on Song et al. instances.

The experiment uses the paper's benchmark protocol: instances with 20, 30,
60, or 90 items, deviation ``a_hat = 0.4 a_bar``, uncertainty budget
``Omega = 0.1 sum(a_hat)``, capacity ``V = (sum(a_bar) + sum(a_hat))/8``,
and overtime coefficient ``c = 1.5/V``.  The input files are under
``data/{30,60,90}`` and are consumed by ``read_instance``.

The driver records the runtime statistics reported in the REBP experiments.
"""

from rebpp import rebppinit_pyomo, print_sol, solve_instance
import numpy as np
import pandas as pd
import math
import time
import statistics as stat
import pyomo.environ as pe
from pyomo.opt import SolverStatus, TerminationCondition

NUM_ITEMS = 90 #20 #90
num_items = [20]
TIME_LIMIT = 7200
devProp = 0.4
c_const = 1.5

num_tests = 10
rob_level_mult = 0.1
V_mult = 1/8



def read_instance(i: int, sz: int, ss: int) -> np.ndarray:
    r"""Load one Song et al. benchmark instance.

    Parameters
    ----------
    i : int
        Instance identifier.
    sz : int
        Benchmark family size, such as 30, 60, or 90.
    ss : int
        Number of rows to retain.

    Returns
    -------
    numpy.ndarray
        Sorted nominal durations ``\bar a``.
    """
    fileName = "../data/" + str(sz) + "/" + str(sz) + "_(1,100)_"+ str(i) +".txt"
    print("opening: ", fileName)
    test_data = pd.read_csv(fileName,skiprows=[1])
    a_bar = np.sort(test_data.iloc[range(ss),0])
    return a_bar

#test_data = pd.read_csv("../data/ma30.csv")
for sz in num_items:
    runTimes = []
    masterTimes = []
    runIter = []
    runBins = []
    runTimesWoTL = []
    masterTimesWoTL = []
    for instNum in range(num_tests):
        a_bar = []
        if sz < 30:
            a_bar = read_instance(instNum, 30, min(NUM_ITEMS, sz))
        else:
            a_bar = read_instance(instNum,sz,min(NUM_ITEMS,sz))

        a_hat = (np.ceil(devProp*a_bar)).astype(int)
        a_bar = np.asarray(a_bar, dtype='int')
        Omega = int(math.ceil(rob_level_mult*sum(a_hat)))
        V = int(V_mult*(sum(a_hat)+sum(a_bar)))
        c_mult = c_const / V

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
