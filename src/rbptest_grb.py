from pyscipopt import Model, quicksum, SCIP_PARAMSETTING
#from knapsack import rebppinit, update_rebpp
from sos2 import sos2, convex_pw_knapsack_dp
from rebpp import rebppinit_pyomo, update_rebpp_pyomo, convex_pw_knapsack_wrapper, print_sol, solve_instance
import numpy as np
import pandas as pd
import math
import time
import statistics as stat
import pyomo.environ as pe
from pyomo.opt import SolverStatus, TerminationCondition



__DEBUG = False
#if __name__ == "__main__":
VIOL_TOL = 1e-3
INT_TOL = 1e-3
TIME_LIMIT = 7200
#3600
devProp = 0.4
c_const = 1.5

num_tests = 10
# Five robustness levels are considered, either 0%, 5%, 10%, 15% or 20% of sum(a_hat)
rob_level_mult = 0.1 #0 #0.05
# Four deadlines are generated for each instance, which are equal to a fraction of the sum of the worst-case job processing times; the fractions considered are 1/4, 1/6,1/8 and 1/10
V_mult = 1/8
#NZ_TOl = 1e-7
#GAPVAL1 = 0.4
#GAPVAL2 = 5e-2



def read_instance(i,sz):
    test_data = pd.read_csv("../data/" + str(sz) + "/" + str(sz) + "_(1,100)_"+ str(i) +".txt",skiprows=[1])
    return test_data.iloc[:,0]

#test_data = pd.read_csv("../data/ma30.csv")
for sz in [60,90]:
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
        a_bar = read_instance(instNum,sz)
        a_hat = (np.ceil(devProp*a_bar)).astype(int)
        a_bar = np.asarray(a_bar, dtype='int')
        Omega = int(math.ceil(rob_level_mult*sum(a_hat)))
        V = int(V_mult*(sum(a_hat)+sum(a_bar)))
        c_mult = c_const / V  # 3/(2*V) #2 / V  # 0.05

        n = len(a_bar)
        m = int(math.ceil(2*(sum(a_bar)+Omega)/V))
        c = np.ones(m)*c_mult
        print("Read file with ", n, " items", " m=", m)
        tmp, timeL, runTime, masterTime, numBins, scenario_num = solve_instance(a_bar, a_hat, V, c,Omega)
        '''alpha = {}
        scenario_num = 0
        opt = pe.SolverFactory('gurobi_direct')
        opt.options['TimeLimit'] = TIME_LIMIT
        gapVal = GAPVAL1
        start = time.time()
        #model, theta, y, f_bar, z = rebppinit_pyomo(a_bar, a_hat, V, c)
        model = rebppinit_pyomo(a_bar, a_hat, V, c)

        it = 0
        numBins = 0
        masterTime = 0
        rTime = 0
        timeL = False
        p_star_old = math.inf
        a_old = []
        #p_old = []
        #b_old = []
        z_old = []

        while True:
            f = {}
            u = {}
            opt.options["MIPGap"] = gapVal
            masterStart = time.time()
            results = opt.solve(model, tee=False)
            masterTime += time.time()-masterStart
            status = results.Solver.status #results.Solver()['Termination condition'].value
            if status != SolverStatus.ok: #TerminationCondition.optimal: #'optimal':
                print('error occurred, status: {status}.  Check model!')
            if results.solver.termination_condition == TerminationCondition.maxTimeLimit:
                timeL = True
                break

            b = np.zeros((m, 3), int)
            p = np.zeros((m, 3), float)
            theta_star = model.theta.value
            numBins = 0
            for j in range(m):
                if model.y[j].value > 1-INT_TOL:
                    f[j] = 0
                    u[j] = 0
                    numBins += 1
                    for i in range(n):
                        # print("loop problem")
                        if model.z[i, j].value > 1-INT_TOL:
                            f[j] += a_bar[i]
                            u[j] += a_hat[i]
                    b[j,1] = max(min(V - f[j],u[j]), 0)
                    b[j,2] = u[j] #,axis=0) #max(u[j] - V + f[j], 0)]]), axis=0)
                    p[j,0] = c[j] * max(f[j] - V, 0)
                    p[j,1] = p[j,0]
                    #p[j,2] = c[j] * max(u[j] - V + f[j], 0)
                    p[j, 2] = c[j] * ( b[j,2]-b[j,1] + max(f[j] - V, 0))
                #p = np.append(p, np.array([[c[j] * max(f[j] - V, 0), c[j] * max(f[j] - V, 0), c[j] * max(u[j] - V + f[j], 0)]]), axis=0)
            #print(b)
            #print(p)
            if __DEBUG:
                print("Before running convex knapsack, Omega=", Omega, " p=", p, " b=", b)
            p_star = 0
            #if np.sum(p[:,2]) > NZ_TOl:
            #p_star_0, a_0 = convex_pw_knapsack_wrapper(p, b, Omega, model.z.extract_values(), a_hat, model, True)
            p_star, a = convex_pw_knapsack_wrapper(p, b, Omega, model.z, a_hat, model, True) #False)
            p_star_0 = p_star

            if p_star_0 != p_star or (p_star == p_star_old and a == a_old and p_star > theta_star + VIOL_TOL):
                print("got same subprob p_star=", p_star, " p_star_old", p_star_old, p_star_0)
                print(a)
                print(a_old)
                #print(a_0)
                print(p)
                print(b)
                #print(p_old)
                #print(b_old)
                print(Omega)
                for k in model.z.keys():
                    if abs(model.z[k].value) > 1e-2:
                        print(model.z[k].getname(), model.z[k].value, end=' ')
                print('')
                print(z_old)
                raise Exception("breaking..")

            p_star_old = p_star
            a_old = a
            z_old=[]
            for k in model.z.keys():
                if abs(model.z[k].value) > 1e-2:
                    z_old.append(model.z[k].getname())
            it += 1
            print("iteration: ", it, " p_star val: ", p_star, " theta_star: ", theta_star, " ******")

            rTime = time.time() - start
            if rTime >= TIME_LIMIT:
                print("time limit")
                break
            elif p_star <= theta_star + VIOL_TOL:
                if gapVal == GAPVAL2:
                    print("terminating, could not find a constraint violating by more than tol=", VIOL_TOL)
                    #print_sol(model)
                    break
                else:
                    gapVal = GAPVAL2
                    print("setting gapVal: ", gapVal)
            model, alpha = update_rebpp_pyomo(model, a_bar, V, c, a, scenario_num)
            scenario_num += 1
        runTime = time.time()-start
        print("Elapsed time instance instNum=", instNum, " elapsed time: ", runTime) '''

        if timeL == False and runTime < TIME_LIMIT-1e-6:
            runTimesWoTL.append(runTime)
            masterTimesWoTL.append(masterTime)
        runIter.append(scenario_num)
        runTimes.append(runTime)
        masterTimes.append(masterTime)
        runBins.append(numBins)

    print(" & {t1:.1f} & {t2:.1f} & {t3:.1f} & {t4:.1f} & {t5:.1f} & {t6:} & {t7:.1f} & {t8:} & {t9:} ".format(t1=stat.mean(runTimes),t2=max(runTimes),t3=stat.mean(masterTimes), t4=max(masterTimes), t5=stat.mean(runIter), t6=max(runIter),t7=stat.mean(runBins), t8=max(runBins), t9=num_tests - len(runTimesWoTL)))
    print(" & ", stat.mean(runTimesWoTL), " & ", max(runTimesWoTL), " & ", stat.mean(masterTimesWoTL), " & ", max(masterTimesWoTL))


#print(" & ", stat.mean(runTimes), " & ", max(runTimes), " & ", stat.mean(runIter), " & ", max(runIter), " & ", stat.mean(runBins), " & ", max(runBins))

#runTimesWoTL = list(filter(lambda x: x<TIME_LIMIT,runTimes))
#masterTimesWoTL = list(filter(lambda x: x<TIME_LIMIT, masterTimes))

      #sum(x >= TIME_LIMIT-0.2 for x in runTimes))

        # model.writeLP("after_update_model.lp")
        # def update_rebpp(model, a_bar, V, c, a, theta, y, f_bar,z):
