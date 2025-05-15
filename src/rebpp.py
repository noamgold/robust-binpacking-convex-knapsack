from copy import deepcopy

from cylp.py.mip.GomoryCutGenerator import epsilon
from fontTools.misc.cython import returns
from numpy.f2py.auxfuncs import throw_error
from pyscipopt import Model, quicksum, SCIP_PARAMSETTING
from sympy import false

#from rbptest import alpha
from sos2 import sos2, convex_pw_knapsack_dp, sos2_gurobi
import numpy as np
import pandas as pd
import pyomo.environ as pe
from pyomo.opt import SolverStatus, TerminationCondition, SolutionStatus
import math
import time

#import os
#os.chdir("c:\\Users\\goldbergno\\My Documents\\GitRepos\\binpacking_summer_project\\src")
FILENAME = "../data/Dep13300with_a_ahat_test_withlabe_V1_HighLoad_01W_020Patients.csv" #"../data/Dep13300with_a_ahat_test_withlabe_V2_HighLoad_04W_071Patients.csv" #"../data/Dep13300with_a_ahat_test_withlabe_V1_HighLoad_02W_032Patients.csv"
    #"../data/Dep13300with_a_ahat_test_withlabe_V1_HighLoad_02W_032Patients.csv"
    #"../data/Dep13300with_a_ahat_test_withlabe_V2_HighLoad.csv"
#"../data/Dep13300with_a_ahat_test_withlabel.csv"
#gapVal = 0.1
"""
sample small test case
"""
__DEBUG = False
__DEBUG_2 = False
__DEBUG_3 = True

#a_hat = [2,2,2,2]
#a_bar = [2,2,3,1]
#Omega = 3 # also B
#V = 8
VIOL_TOL = 1e-6
INT_TOL = 1e-2
GAPVAL1 = 0.4  # optimality gap to finish 1st phase of algorithm
GAPVAL2 = 0.01 # final optimality gap
TIME_LIMIT = 10800
#3600
Omega = 0 #1195 #1833 #6692 #3551 #6692 #3551 #2523 #1833  # 3000 #240 # also B

MAX_BINS = 5
MAX_SCENRIOS = 1e4

SOS_SOLVE = False
BINSIZE = 720
OT_cost = 1.5/BINSIZE #0.0021 #0.003

EPS = 10**-10

import os  # change current path to the file's directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def getVal(model, var):
        return var.value

def getVars(model):
    return model.component_data_objects(model, pe.Var,active=False)

#first fit decreasing heuristic for robust bin packing with omega-uncertainty
def FFD_RBPP(a_bar, a_hat, V, Omega_in):
    """
    first fit decreasing
    """
    nominal_fill = [0]
    deviation_fill = [0]
    sol = [[]]
    a_bar_array = np.array(a_bar)
    a_hat_array = np.array(a_hat)
    ratios = np.divide(a_hat_array,a_bar_array)
    indexes = np.argsort(ratios)
    reversed_indexes = np.flip(indexes)

    for index in reversed_indexes:
        for j in range(len(nominal_fill)):
            if nominal_fill[j] + min(Omega_in, deviation_fill[j] + a_hat[index]) + a_bar[index] <= V:
                nominal_fill[j] += a_bar[index]
                deviation_fill[j] += min(a_hat[index], Omega_in - deviation_fill[j])
                sol[j].append(index)
                break
            else:
                sol.append([index])
                nominal_fill.append(a_bar[index])
                deviation_fill.append(min(a_hat[index], Omega_in))
    return sol


U = {}

# robust extensible bin packing problem model init
def rebppinit_pyomo(a_bar, a_hat, V, c):
    m = len(c)
    n = len(a_bar)
    mdl = pe.ConcreteModel()
    pe.ConcreteModel.getVal = classmethod(getVal)
    pe.ConcreteModel.getVars = classmethod(getVars)

    mdl.I = range(n)
    mdl.J = range(m)
    mdl.theta = pe.Var(domain=pe.NonNegativeReals)
    #mdl.fover = pe.Var(mdl.J,domain=pe.NonNegativeReals)

    mdl.y = pe.Var(mdl.J,domain=pe.Binary)
    mdl.z = pe.Var(mdl.I,mdl.J,domain=pe.Binary)

    mdl.sn = pe.Param(initialize = 1,domain=pe.NonNegativeIntegers,mutable=True)
    #mdl.scenarios = pe.Set(initialize=mdl.sn[])
    #mdl.alpha_bar = pe.Var(mdl.J,pe.NonNegativeReals)
    mdl.alpha_bar = pe.Var(mdl.J,domain=pe.NonNegativeReals,dense=False)
    mdl.alpha = pe.Var(mdl.J,pe.NonNegativeIntegers,domain=pe.NonNegativeReals,dense=False)

    def assignRule(mdl,i):
        return sum(mdl.z[i, j] for j in mdl.J) == 1
    mdl.assignCons = pe.Constraint(mdl.I,rule=assignRule)

    def capacityRule(mdl, j):
        return sum(a_bar[i]*mdl.z[i, j] for i in mdl.I) <= V*mdl.y[j]+mdl.alpha_bar[j]
    mdl.capacityCons = pe.Constraint(mdl.J, rule=capacityRule)

    def indRule(mdl,i,j):
        return (mdl.z[i,j] <= mdl.y[j])
    mdl.indCons = pe.Constraint(mdl.I,mdl.J,rule=indRule)

    #def overFlowRule(mdl, j):
    #    return (mdl.fover[j]<=mdl.alpha_bar[j])
    #mdl.overCons = pe.Constraint(mdl.J,rule = overFlowRule)   ## added new 28.4.25 fover variable to make cuts continuous

    #def indRuleB(mdl,j):
    #    return (mdl.alpha_bar[j] <= 2*V*mdl.y[j])    #### experiment with bound on overfilling - can add this for each scenario as well
    #mdl.indConsB = pe.Constraint(mdl.J,rule=indRuleB)

    #def symbreak(mdl, j):
    #    return (mdl.y[j] >= mdl.y[j+1])
    #mdl.symCons = pe.Constraint(mdl.J[:-1],rule=symbreak)
    mdl.objCons = pe.Constraint(expr = sum(c[j]*mdl.alpha_bar[j] for j in mdl.J)<= mdl.theta)
    mdl.obj = pe.Objective(expr = sum(mdl.y[j] for j in mdl.J) + mdl.theta, sense=pe.minimize)
    mdl.scuts = pe.ConstraintList()

    return mdl #, mdl.theta, mdl.y, mdl.alpha_bar, mdl.z



# robust extensible bin packing problem model init
def rebppinit(a_bar, a_hat, V, c):
    model = Model("rebpp")
    m = len(c)
    n = len(a_bar)

    y, alpha_bar, z = {}, {}, {}
    theta = model.addVar(vtype="C", name="theta")

    # initialize variables
    for j in range(m):
        y[j] = model.addVar(vtype="B", name="y(%s)" % j)
        alpha_bar[j] = model.addVar(vtype="C", name="alpha_bar(%s)" % j)
        for i in range(n):
            z[i, j] = model.addVar(vtype="B", name="z(%s,%s)" % (i, j))

    # initialize constraints
    for i in range(n):
        model.addCons(quicksum(z[i, j] for j in range(m)) == 1, "Assign(%s)" % i)  # constraint 1b
        for j in range(m):
            model.addCons(z[i, j] <= y[j], "Strong(%s,%s)" % (i, j))  # constraint 1c

    for j in range(m):
        model.addCons(
            quicksum(a_bar[i] * z[i, j] for i in range(n)) <= alpha_bar[j] + y[j] * V)  # moved y[j] * V to other side

    model.addCons(quicksum(c[j] * alpha_bar[j] for j in range(m)) <= theta)

    model.setObjective(quicksum(y[j] for j in range(m)) + theta, "minimize")

    # model.writeLP("initial_model.lp")

    return model, theta, y, alpha_bar, z


def update_rebpp_pyomo(mdl, a_bar, V, c, a, scenario_num):
    """
    used to recieve new model and alph
    """
    if __DEBUG_2:
        print("scenario_num: ", scenario_num)

    #mdl.scenarios = mdl.scenarios | pe.Set(initialize=[scenario_num])
    mdl.sn = scenario_num
    for j in mdl.J:
        mdl.scuts.add(sum(mdl.z[i, j] * (a_bar[i] + a[i]) for i in mdl.I) <= V * mdl.y[j] + mdl.alpha[j, scenario_num])
        #mdl.scuts.add(mdl.fover[j] <= mdl.alpha[j, scenario_num])
    mdl.scuts.add(sum(c[j] * mdl.alpha[j, scenario_num] for j in mdl.J) <= mdl.theta)

    return mdl, mdl.alpha


def update_rebpp(model, a_bar, V, c, a, theta, y, z, alpha, scenario_num):
    """
    used to recieve new model and alph
    """
    m = len(y)
    n = len(a_bar)
    model.freeTransform()
    if __DEBUG_2:
        print("scenario_num: ", scenario_num)
    for j in range(m):
        alpha[j, scenario_num] = model.addVar(vtype="C", name="alpha(%s,%s)" % (j, scenario_num))
        model.addCons(quicksum(z[i, j] * (a_bar[i] + a[i]) for i in range(n)) <= V * y[j] + alpha[j, scenario_num])

    model.addCons(quicksum(c[j] * alpha[j, scenario_num] for j in range(m)) <= theta)

    return model, alpha


def convex_pw_knapsack_wrapper(p, b, Omega, z, a_hat, model, sos = True):
    """
    used to retrive the p_star and a values
    """
    n = len(a_hat)
    m = len(p)
    a = [0] * n
    p_star = None
    items = []
    i_max = None
    # if we use the scip version

    m,nn = p.shape # n = rows // m = columns
    p_star = 0
    p_star_k = []
    pp = np.array(p)
    bb = np.array(b)
    constant = 0
    #itemsConstant = set([])
    for i in range(m):
        if not all(p[i, j] <= p[i, j + 1] for j in range(nn - 1)):
            print(p, b)
            raise ValueError("nonconvex p")
        if not all(b[i, j] <= b[i, j + 1] for j in range(nn - 1)):
            print(p, b)
            raise ValueError("nonconvex b")
        if p[i, 0] > INT_TOL:  #and not sos:
            pp[i, 0] = 0
            pp[i, 1] = 0
            pp[i,2] -= p[i,0]
            constant += p[i,0]
            #itemsConstant.add(i)  not needed
    if sos:
        if __DEBUG_3:
            p_star_k, items_k, i_max_k = convex_pw_knapsack_dp(pp, bb, Omega, False)  # true
            print("p_star knapsack = ", p_star_k, " items_k=", items_k, " i_max_k=", i_max_k, " constant=", constant)
        p_star, items, i_max, _, _ = sos2_gurobi(pp, bb, Omega)
        print("p_star sos = ", p_star, " items=", items, " i_max=", i_max)

    else:
        if Omega > 0:
            p_star, items, i_max = convex_pw_knapsack_dp(pp,bb,Omega) #,True) # true since y intercept is nonzero
    #itemsConstant = itemsConstant.difference([i_max])
    #items = itemsConstant.union(items)
    fullDevSum = 0
    for item in items:
        for i in range(n):
            if model.getVal(z[i,item]) >= 1 - INT_TOL:
                a[i] = a_hat[i]
                fullDevSum += a_hat[i]
    remDev = Omega - fullDevSum
    if remDev < 0:
        print ("Error", Omega, remDev, items, i_max)
        raise ValueError
    if i_max is not None:
        for i in range(n):
            if model.getVal(z[i,i_max]) >= 1 - INT_TOL:
                a[i] = min(a_hat[i],remDev)
                remDev -= a[i]
    if p_star > 0 and remDev > 0:
        if __DEBUG_2 or __DEBUG_3:
            p_star_2 = []
            items_2 = []
            i_max_2 = []
            if not sos:
                p_star_2, items_2, i_max_2, _, _ = sos2_gurobi(p, b, Omega)
            else:
                p_star_2 = p_star_k
                items_2 = items_k
                i_max_2 = i_max_k
            print ("items=",items, " i_max=", i_max, " remDev=", remDev, " sumU=", sum(b[i,2] for i in range(m)), " sum b[:,1]=", sum(b[i,1] for i in range(m)),
                   " Omega=", Omega, " p_star=", p_star, " p_star_2=", p_star_2, " constant=", constant, " items_2", items_2, " i_max_2=", i_max_2)
            if (not sos and abs(p_star_2 - (p_star+constant)) > 1e-3) or (sos and abs(p_star_2+constant - p_star) > 1e-3):
                #print("p_star sos = ", p_star, " items=", items, " i_max=", i_max, " p_star_2=", p_star_2, " i_max_2=", i_max_2, " items_2=", items_2, " constant=", constant)
                print(p, b)
                raise ValueError("error in conv knapsack")
            #raise ValueError("remDev>0")
    return p_star+constant, a

def print_sol(model):
    """
    used to print the values of the variables and the objective value
    """
    for var in model.getVars():
        val = model.getVal(var)
        if val != 0:
            print(var,":",model.getVal(var), end=" ")
    print("\nmodel obj val: ",model.getObjVal())


def solve_instance(a_bar, a_hat, V, c, Omega, timelimit = TIME_LIMIT):
    n = len(a_bar)
    m = len(c) #int(math.ceil(2 * (sum(a_bar) + Omega) / V))
    print("Read file with ", n, " items", " m=", m)

    alpha = {}
    scenario_num = 0
    opt = pe.SolverFactory('gurobi_direct')
    gapVal = GAPVAL1
    start = time.time()
    # model, theta, y, f_bar, z = rebppinit_pyomo(a_bar, a_hat, V, c)
    model = rebppinit_pyomo(a_bar, a_hat, V, c)

    it = 0
    numBins = 0
    masterTime = 0
    timeL = False
    p_star_old = math.inf
    a_old = []
    z_old = []
    model_old = []
    while True:
        f = {}
        u = {}
        opt.options["MIPGap"] = gapVal
        opt.options['TimeLimit'] = int(float(timelimit)-(time.time()-start))
        masterStart = time.time()
        results = opt.solve(model, tee=False)
        masterTime += time.time() - masterStart
        soln = results.Solution
        status = results.Solver.status  # results.Solver()['Termination condition'].value
        if results.solver.termination_condition == TerminationCondition.maxTimeLimit:
            print('time limit!')
            timeL = True
            break
        elif status != SolverStatus.ok:  # TerminationCondition.optimal: #'optimal':
            error('error occurred, status: {}.  Check model!'.format(status))

        b = np.zeros((m, 3), int)
        p = np.zeros((m, 3), float)
        theta_star = model.theta.value
        numBins = 0
        for j in range(m):
            if model.y[j].value > 1 - INT_TOL:
                f[j] = 0
                u[j] = 0
                numBins += 1
                for i in range(n):
                    # print("loop problem")
                    if model.z[i, j].value > 1 - INT_TOL:
                        f[j] += a_bar[i]
                        u[j] += a_hat[i]
                b[j, 1] = max(min(V - f[j], u[j]), 0)
                b[j, 2] = u[j]  # ,axis=0) #max(u[j] - V + f[j], 0)]]), axis=0)
                p[j, 0] = c[j] * max(f[j] - V, 0)
                p[j, 1] = c[j] * max(f[j] - V, 0)
                # p[j,2] = c[j] * max(u[j] - V + f[j], 0)
                p[j, 2] = c[j] * (b[j, 2] - b[j, 1] + max(f[j] - V, 0))
                if abs(p[j,2])<=1e-3:  # if b[j,1]=u[j] < f[j] - V
                    b[j,1] = 0
                    b[j,2] = 0
                assert p[j,2] >= p[j,1]
                assert p[j,1] >= p[j,0]
            # p = np.append(p, np.array([[c[j] * max(f[j] - V, 0), c[j] * max(f[j] - V, 0), c[j] * max(u[j] - V + f[j], 0)]]), axis=0)
        # print(b)
        # print(p)
        if __DEBUG:
            print("Before running convex knapsack, Omega=", Omega, " p=", p, " b=", b)
        p_star = 0
        # if np.sum(p[:,2]) > NZ_TOl:
        # p_star_0, a_0 = convex_pw_knapsack_wrapper(p, b, Omega, model.z.extract_values(), a_hat, model, True)
        p_star, a = convex_pw_knapsack_wrapper(p, b, Omega, model.z, a_hat, model, SOS_SOLVE)  # False)  # if last argument is false then DP is invoked otherwise SoS
        p_star_0 = p_star

        if p_star_0 != p_star or (p_star == p_star_old and a == a_old and p_star > theta_star + 100*VIOL_TOL):
            print("got same subprob p_star=", p_star, " p_star_old", p_star_old, p_star_0)
            print(a)
            print(a_old)
            print(p)
            print(b)
            print(Omega)
            for k in model.z.keys():
                if abs(model.z[k].value) > 1e-2:
                    print(model.z[k].getname(), model.z[k].value, end=' ')
            print('')
            print(z_old)
            model.scuts.pprint()
            model_old.scuts.pprint()
            raise Exception("breaking..")

        p_star_old = p_star
        a_old = a
        model_old = deepcopy(model)
        z_old = []
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
                # print_sol(model)
                break
            else:
                gapVal = GAPVAL2
                print("setting gapVal: ", gapVal)
        model, alpha = update_rebpp_pyomo(model, a_bar, V, c, a, scenario_num)
        scenario_num += 1
    runTime = time.time() - start
    print(" elapsed time: ", runTime)
    assign = pd.DataFrame(columns=["patient", "shift"])
    for k in model.z.keys():
        if abs(model.z[k].value) > 1e-2:
            assign = pd.concat([assign,pd.DataFrame(np.array([[k[0], k[1]]]), columns=assign.columns)],ignore_index=True)
    return assign, timeL, runTime, masterTime, numBins, scenario_num

if __name__ == "__main__":
    # example problem

    pe.ConcreteModel.getVal = classmethod(getVal)

    a_hat = []
    V = BINSIZE
    BEGIN = 0
    rambam_data = pd.read_csv(FILENAME)
    a_bar = rambam_data["a"]
    a_hat = rambam_data["ahat"]
    END = len(a_bar)

    a_bar = np.round(a_bar[BEGIN:END].to_numpy())
    a_hat = np.round(a_hat[BEGIN:END].to_numpy())
    a_bar = np.asarray(a_bar, dtype = 'int')
    a_hat = np.asarray(a_hat, dtype = 'int')
    m = MAX_BINS #int(math.ceil(2 * (sum(a_bar) + Omega) / V))
    c = [OT_cost]*m #[0.003,0.003,0.003,0.003,0.003,0.003,0.003,0.003]
    c += EPS*np.array(range(m))

    assign,timeL, runTime, masterTime, numBins, scenario_num = solve_instance(a_bar, a_hat, V, c,Omega)
    print(" time limit: ", timeL, "run time: ", runTime, " master runtime: ", masterTime, " num of bins: ", numBins, " num of scenarios: ", scenario_num)
    #error("quit")
    print(assign)
    assign.to_csv(FILENAME + "_schedule.csv")

"""    m = len(c)
    n = len(a_bar)
    alpha = {}
    scenario_num = 0
    #model, theta, y, f_bar, z = rebppinit(a_bar,a_hat,V,c)
    model = rebppinit_pyomo(a_bar,a_hat,V,c)
    theta = model.theta
    #model.hideOutput()
    it = 0
    opt = pe.SolverFactory('gurobi_direct')
    p_star_old = 0
    a_old = []
    z_old = []

    while True:
        f = {}
        u = {}
        #model.optimize()
        opt.options["MIPGap"] = gapVal
        results = opt.solve(model, tee=False)

        #print_sol(model)
        #model.writeProblem("model" + str(iter) + ".cip",trans=False)
        # model = model2
        b = np.zeros((m, 3), int)  #empty((0,3), int)
        p = np.zeros((m, 3), float)  #empty((0,3), int)
#        if model.getStatus() != "optimal":
        status = results.Solver.status  # results.Solver()['Termination condition'].value
        if status != SolverStatus.ok:  # TerminationCondition.optimal: #'optimal':
            print("Error (suboptimal)")
            raise ValueError

        for j in range(m):
            if model.getVal(model.y[j]) > 1 - INT_TOL:
                f[j] = 0
                u[j] = 0
#                numBins += 1
                for i in range(n):
                    # print("loop problem")
                    if model.getVal(model.z[i, j]) > 1 - INT_TOL:
                        f[j] += a_bar[i]
                        u[j] += a_hat[i]
                b[j, 1] = max(min(V - f[j], u[j]), 0)
                b[j, 2] = u[j]  # ,axis=0) #max(u[j] - V + f[j], 0)]]), axis=0)
                p[j, 0] = c[j] * max(f[j] - V, 0)
                p[j, 1] = p[j, 0]
                p[j, 2] = c[j] * (b[j, 2] - b[j, 1] + max(f[j] - V, 0))
                assert p[j,2] >= p[j,1]
                assert p[j,1] >= p[j,0]
#            f[j] = 0
#            u[j] = 0
#            for i in range(n):
#                if model.getVal(z[i,j]) > 1 - INT_TOL:
#                    f[j] += a_bar[i]
#                    u[j] += a_hat[i]
#            b = np.append(b, np.array([[0, min(max(V-f[j],0),u[j]),u[j]]]),axis=0)
#                                        #max(u[j]-V+f[j],0)]]), axis=0)   # 29/3 - added u[j] truncation in 2nd breakpoint
#            p = np.append(p, np.array([[c[j]*max(f[j]-V,0), c[j]*max(f[j]-V,0), c[j]*max(u[j]-V+f[j],0)]]), axis=0)
        #print(b)
        #print(p)
        theta_star = model.getVal(theta)
        #print(model.z.extract_values())

        p_star,a = convex_pw_knapsack_wrapper(p,b,Omega,model.z,a_hat,model,False)
        it += 1
        print("iteration: ", it, " p_star val: ", p_star, " theta_star_val: ",theta_star)
        # p_star_0, a_0 = convex_pw_knapsack_wrapper(p, b, Omega, model.z.extract_values(), a_hat, model, True)

        p_star_0, a_0 = convex_pw_knapsack_wrapper(p, b, Omega, model.z, a_hat, model, True)  # False)

        if p_star_0 != p_star or (p_star == p_star_old and a == a_old and p_star > theta_star + VIOL_TOL):
            print("got same subprob p_star=", p_star, " p_star_old", p_star_old, p_star_0)
            print(a)
            print(a_old)
            # print(a_0)
            print(p)
            assert p[0,1]>=p[0,0]
            print(b)
            # print(p_old)
            # print(b_old)
            print(Omega)
            for k in model.z.keys():
                if abs(model.z[k].value) > 1e-2:
                    print(model.z[k].getname(), model.z[k].value, end=' ')
            print('')
            print(z_old)
            raise Exception("breaking..")

        p_star_old = p_star
        a_old = a
        z_old = model.z.extract_values()
        if p_star <= theta_star + VIOL_TOL:
            print_sol(model)
            break
        #model,alpha = update_rebpp(model, a_bar, V, c, a, theta, y, z, alpha, scenario_num)
        model, alpha = update_rebpp_pyomo(model, a_bar, V, c, a,scenario_num)
        scenario_num += 1
        # model.writeLP("after_update_model.lp")
        # def update_rebpp(model, a_bar, V, c, a, theta, y, f_bar,z):
"""
