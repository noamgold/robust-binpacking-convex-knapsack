r"""Robust extensible bin-packing (REBP) master and separation workflow.

Paper-to-code notation:
    ``a_bar`` is the nominal duration vector ``\bar a``; ``a_hat`` is the
    deviation vector ``\hat a``; ``V`` is the shift capacity; ``Omega`` is the
    budget uncertainty parameter; ``c`` contains overtime costs; ``y`` and
    ``z`` are the master bin-opening and assignment variables; and ``theta``
    is the worst-case overtime objective component.  ``solve_instance``
    implements Algorithm 1 (row-and-column generation), while
    ``convex_pw_knapsack_wrapper`` calls the two-piece convex-knapsack
    separation method from ``sos2.py``.

The Pyomo model corresponds to formulation (1).  Generated scenarios are
added through ``add_cut``/``update_rebpp_pyomo`` until the separation value
``eta*`` is within ``VIOL_TOL`` of ``theta``.
"""

import os
import sys
env_name = os.environ.get('CONDA_DEFAULT_ENV')
print(f"Active Conda Environment Name: {env_name}")


from copy import deepcopy
from gurobipy import GRB
from sos2 import sos2, convex_pw_knapsack_omega_dp, sos2_gurobi
import numpy as np
import pandas as pd
import pyomo.environ as pe
from pyomo.opt import SolverStatus, TerminationCondition, SolutionStatus
import math
import time
from numba import jit
from typing import Any, List, Tuple

VALIDINEQ2 = False  # inequalities added dynamically
VALIDINEQ = True  # m inequalities added during initialization
SYMBREAK = True  # symmetry breaking by ordering bins
ITEMSYMBREAK = False  # symmetry breaking by ordering equal-sized bins
BRANCH_AND_CUT = False
NO_VAR_GEN = False
#
SOS_SOLVE = False
MIP_START_OR_HINT = 2  # 2: hint, 1: start, 0: none

# Solver tolerances and algorithm controls.
# TAU0 and TAU1 are hard-coded paper parameters; change them manually to
# reproduce the paper's alternative two-phase optimality-gap configurations.

VIOL_TOL = 1e-3
INT_TOL = 1e-1
TAU0 = 0.2
TAU1 = 0.01
TIME_LIMIT = 7200
MAX_CUTS = math.inf
#################################
DEBUG_CB = False
DEBUG_CB_0 = False
DEBUG_CB_2 = False
GUROBI_OUTPUT = False
DEBUG_INEQ_NOVAR = False
BOUND_OVERFILL = math.inf  # 2*BINSIZE
EPS = 1e-6

FILENAME = "../data/Dep13300with_a_ahat_test_withlabe_V1_HighLoad_01W_V12.csv"
Omega_val = 971
BINSIZE = 540

__DEBUG = False
__DEBUG_0 = False
__DEBUG_2 = False
__DEBUG_3 = False
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def getVal(model: Any, var: Any) -> Any:
    """Return a Pyomo variable value for legacy model adapters."""
    return var.value

def getVars(model: Any) -> Any:
    """Return active Pyomo variables from a model instance."""
    return model.component_data_objects(pe.Var,active=True)

def rebppinit_pyomo(
    a_bar: np.ndarray,
    a_hat: np.ndarray,
    V: float,
    c: np.ndarray,
    Omega: float,
) -> Any:
    r"""Create the finite-scenario Pyomo master for formulation (1).

    The arrays correspond to ``\bar a``, ``\hat a``, and the fixed model
    parameters in the paper; ``c`` supplies the per-bin overtime costs.
    """
    m = len(c)
    n = len(a_bar)

    b = np.diff(a_bar)
    min_abar_diff = b[b > 0].min()
    model = pe.ConcreteModel()
    pe.ConcreteModel.getVal = classmethod(getVal)
    pe.ConcreteModel.getVars = classmethod(getVars)

    model.I = range(n)
    model.J = range(m)
    model.a_bar = a_bar
    model.a_hat = a_hat
    model.V = V
    model.c = c
    model.Omega = Omega
    model.theta = pe.Var(domain=pe.NonNegativeReals)
    model.y = pe.Var(model.J,domain=pe.Binary)
    model.z = pe.Var(model.I,model.J,domain=pe.Binary)

    model.sn = pe.Param(initialize = 1,domain=pe.NonNegativeIntegers,mutable=True)
    #mdl.scenarios = pe.Set(initialize=mdl.sn[])
    model.alpha_bar = pe.Var(model.J,domain=pe.NonNegativeReals,dense=False, bounds=(0,BOUND_OVERFILL))
    model.alpha = pe.Var(model.J,pe.NonNegativeIntegers,domain=pe.NonNegativeReals,bounds=(0,BOUND_OVERFILL),dense=False)

    def assignRule(mdl,i):
        # Eq. (1b): each item receives exactly one bin assignment.
        return sum(mdl.z[i, j] for j in mdl.J) == 1
    model.assignCons = pe.Constraint(model.I,rule=assignRule)

    def capacityRule(mdl, j):
        # Eq. (1d): nominal load plus scenario deviation is extensible.
        return sum(a_bar[i]*mdl.z[i, j] for i in mdl.I) <= V*mdl.y[j]+mdl.alpha_bar[j]
    model.capacityCons = pe.Constraint(model.J, rule=capacityRule)

    def indRule(mdl,i,j):
        # Eq. (1c): assignments imply an opened bin.
        return mdl.z[i,j] <= mdl.y[j]
    model.indCons = pe.Constraint(model.I,model.J,rule=indRule)

    #def overFlowRule(mdl, j):
    #    return (mdl.fover[j]<=mdl.alpha_bar[j])
    #mdl.overCons = pe.Constraint(mdl.J,rule = overFlowRule)   ## added new 28.4.25 fover variable to make cuts continuous
    #def indRuleB(mdl,j):
    #    return (mdl.alpha_bar[j] <= 2*V*mdl.y[j])    #### experiment with bound on overfilling - can add this for each scenario as well
    #mdl.indConsB = pe.Constraint(mdl.J,rule=indRuleB)
    model.sumA = sum(model.a_bar[i] for i in model.I)

    def symbreak(mdl, j):
        return mdl.y[j] >= mdl.y[j+1]

    def itemSymBreak(mdl, i, j):
        #return (mdl.z[i,j] - mdl.z[i+1,j] <= (a_bar[i+1]-a_bar[i])/min_abar_diff)
        if i < j:
            return pe.Constraint.Skip
        #return sum(mdl.z[i,p] for p in range(j,min(m,i+1))) <= sum(mdl.z[p,j-1] for p in range(i))
        return (sum(mdl.z[i,p] for p in range(j,min(m,i+1))) <= sum(mdl.z[p,j-1] for p in range(j-1,i)))

    def itemSymBreakB(mdl, i):
        return sum(mdl.z[i,j] for j in range(i+1)) == 1

    def overtimeLB(mdl, j):
        ot = mdl.sumA+mdl.Omega-(j+1)*mdl.V
        if ot <= 0:
            return pe.Constraint.Skip
        return (1-mdl.y[j+1])*mdl.c[0]*ot <= mdl.theta
    #((mdl.y[j]-mdl.y[j+1])*(mdl.c[0]*(mdl.sumA+mdl.Omega-(j+1)*mdl.V)) <= mdl.theta)

    if ITEMSYMBREAK:
        model.symConsItemA = pe.Constraint(range(1,n),range(1,m), rule=itemSymBreak)
        model.symConsItemB = pe.Constraint(model.J, rule=itemSymBreakB)         #pe.Constraint(model.I[:-1],model.J, rule=itemSymBreak))

    if SYMBREAK or VALIDINEQ:
        model.symCons = pe.Constraint(model.J[:-1],rule=symbreak)
        if VALIDINEQ:  # add only if not generating these dynamically
            model.otLB = pe.Constraint(model.J[:-1],rule=overtimeLB)

    # Eq. (1e): aggregate overtime is bounded by Theta.
    model.objCons = pe.Constraint(expr = sum(c[j]*model.alpha_bar[j] for j in model.J)<= model.theta)

    # Eq. (1a): minimize opened bins plus worst-case overtime.
    model.obj = pe.Objective(expr = (sum(model.y[j] for j in model.J) + model.theta), sense=pe.minimize)
    ##mdl.obj = pe.Objective(expr = (sum(mdl.y[j] for j in mdl.J) + sum(c[j]*mdl.alpha_bar[j] for j in mdl.J) + mdl.theta), sense=pe.minimize)
    model.scuts = pe.ConstraintList()
    model.cuts_added = 0

    return model #mdl #, mdl.theta, mdl.y, mdl.alpha_bar, mdl.z


def add_cut(mdl: Any, a_bar: np.ndarray, a: np.ndarray) -> None:
    r"""Add a scenario cut for a candidate scenario ``a \in U_\Omega``."""
    sum_expression = 0
    m = len(a_bar)
    for j in mdl.J:
        binItems = []
        filledCapacity = 0
        if mdl.y[j].value > 1 - INT_TOL:
            for i in range(m):
                if mdl.z[i, j].value >= 1 - INT_TOL:
                    binItems.append(i)
                    filledCapacity += a_bar[i] + a[i]
        if filledCapacity > V:
            sum_expression -= c[j] * V * mdl.y[j]
            for i in binItems:
                sum_expression += c[j] * (a_bar[i] + a[i]) * mdl.z[i, j]
    cons = mdl.scuts.add(sum_expression <= mdl.theta)
    if DEBUG_INEQ_NOVAR:
        print("found cut, theta: ", mdl.theta.value)
        cons.pprint()



def update_rebpp_pyomo(
    mdl: Any,
    a_bar: np.ndarray,
    V: float,
    c: np.ndarray,
    a: np.ndarray,
    scenario_num: int,
    no_var: bool = NO_VAR_GEN,
) -> Tuple[Any, Any]:
    r"""Add scenario-indexed constraints to the finite master model."""
    if __DEBUG_2:
        print("scenario_num: ", scenario_num)
    m = len(a_bar)
    mdl.sn = scenario_num
    if no_var:
        add_cut(mdl,a_bar,a)
    else:
        for j in mdl.J:
            cons = mdl.scuts.add(sum(mdl.z[i, j] * (a_bar[i] + a[i]) for i in mdl.I) <= V * mdl.y[j] + mdl.alpha[j, scenario_num])
        con = mdl.scuts.add(sum(c[j] * mdl.alpha[j, scenario_num] for j in mdl.J) <= mdl.theta)
    return mdl, mdl.alpha

def convex_pw_knapsack_wrapper(
    p: np.ndarray,
    b: np.ndarray,
    Omega: float,
    z: np.ndarray,
    a_hat: np.ndarray,
    sos: bool = True,
) -> Tuple[float, List[float], float]:
    r"""Solve the separation problem and recover a worst-case scenario.

    The inputs encode Eq. (3)'s ``gamma_j``, ``beta_j``, and ``u_j`` for the
    current master assignment ``Z^*``. The return value is ``eta^*`` from
    Proposition 1, the deviation vector ``a^* \in U_\Omega``, and any fixed
    intercept contribution.
    """
    n = len(a_hat)
    m = len(p)
    a = [0] * n
    p_star = None
    items = []
    i_max = -1 #None
    m, nn = p.shape
    p_star = 0
    p_star_k = []
    pp = np.array(p)
    bb = np.array(b)
                # itemsConstant.add(i)  not needed
    constant = 0
    for i in range(m):
        if not np.all(p[i, j] <= p[i, j + 1] for j in range(nn - 1)):
            print(p, b)
            raise ValueError("nonconvex p")
        if not np.all(b[i, j] <= b[i, j + 1] for j in range(nn - 1)):
        # itemsConstant = itemsConstant.difference([i_max])
        # items = itemsConstant.union(items)
            raise ValueError("nonconvex b")
        if p[i, 0] > INT_TOL:  #and not sos:
            pp[i,2] = max(pp[i,2]-p[i,0],0)
                # print("p_star sos = ", p_star, " items=",items, " i_max=", i_max, " p_star_2=", p_star_2, " i_max_2=", i_max_2, " constant=", constant)
            pp[i,0] = 0
            pp[i,1] = 0
            constant += p[i,0]

            #itemsConstant.add(i)  not needed
    if sos:
        if __DEBUG_3:
            p_star_k, items_k, i_max_k = convex_pw_knapsack_omega_dp(pp, bb, Omega, False)  # true
            print("p_star knapsack = ", p_star_k, " items_k=", items_k, " i_max_k=", i_max_k, " constant=", constant)
        _star, items, i_max, _, _ = sos2_gurobi(pp, bb, Omega)
        if __DEBUG_0:
            print("p_star sos = ", p_star, " items=", items, " i_max=", i_max)

    else:
        if Omega > 0:
            p_star, items, i_max = convex_pw_knapsack_omega_dp(pp,bb,Omega) #,True) # true since y intercept is nonzero
    #itemsConstant = itemsConstant.difference([i_max])
    #items = itemsConstant.union(items)
    fullDevSum = 0
    for item in items:
        for i in range(n):
            if z[i,item] >= 1 - INT_TOL:
                a[i] = a_hat[i]
                fullDevSum += a_hat[i]
    remDev = Omega - fullDevSum
    if remDev < 0:
        print ("Error", Omega, remDev, items, i_max)
        raise ValueError
    if i_max != -1: #None:
        for i in range(n):
            if z[i,i_max] >= 1 - INT_TOL:
                a[i] = min(a_hat[i],remDev)
                remDev -= a[i]
    if p_star > 0 and remDev > INT_TOL:
        if __DEBUG_2 or __DEBUG_3:
            p_star_2 = []
            items_2 = []
            i_max_2 = []
            if not sos:
                p_star_2, items_2, i_max_2, _, _ = sos2_gurobi(pp, bb, Omega)
            else:
                p_star_2 = p_star_k
                #items_2 = items_k
                #i_max_2 = i_max_k
            #print ("items=",items, " i_max=", i_max, " remDev=", remDev, " sumU=", sum(b[i,2] for i in range(m)), " sum b[:,1]=", sum(b[i,1] for i in range(m)),
            #       " Omega=", Omega, " p_star=", p_star, " p_star_2=", p_star_2, " constant=", constant, " items_2", items_2, " i_max_2=", i_max_2)
            if (not sos and abs(p_star_2 - (p_star+constant)) > 1e-3) or (sos and p_star_k and abs(p_star_2- p_star) > 1e-3):
                #print("p_star sos = ", p_star, " items=", items, " i_max=", i_max, " p_star_2=", p_star_2, " i_max_2=", i_max_2, " items_2=", items_2, " constant=", constant)
                print(p, b)
                raise ValueError("error in conv knapsack")

        #raise ValueError("remDev>0 "+str(remDev) +" " +str(i_max))
    return p_star+constant, a, constant

#model = pe.ConcreteModel()

#@jit(nopython=True)
def callback_helper(mmodel: Any, model: Any) -> np.ndarray:
    r"""Extract an integral assignment ``Z^*`` from a Gurobi callback."""
    yy = []
    #zz = []
    n = len(model.a_bar)
    m = len(model.c)
    Z = np.zeros([n, m])

    for j in model.J:
        #y = mmodel.getVarByName("y(" + str(j) + ")")
        #yval = mmodel.cbGetSolution(y)
        #if yval > 1 - INT_TOL:
        #    yy.append(j)
        #elif yval > INT_TOL:
        #    print("ERR: non-integer soln")
        for i in model.I:
            z = mmodel.getVarByName("z(" + str(i) + "_" + str(j) + ")")
            zval = mmodel.cbGetSolution(z)
            if zval > 1 - INT_TOL:
                #zz.append((i, j))
                Z[i,j]=1
    #idxs = np.array(zz)
    #Z[idxs[:, 0], idxs[:, 1]] = 1
    return Z

opt = pe.SolverFactory('gurobi_persistent', report_timing=True)

def add_cut_two(
    Z: np.ndarray,
    a_bar: np.ndarray,
    a: np.ndarray,
    V: float,
    c: np.ndarray,
    model: Any,
) -> Any:
    r"""Add a lazy robust cut for assignment matrix ``Z^*``."""
    fill = (a_bar+a)@Z
    JJ = np.where(fill > V)
    if DEBUG_CB:
        print(fill, [j.item() for j in np.nditer(JJ)])
    sum_expression = 0
    for j in np.nditer(JJ):
        sum_expression = sum_expression - c[j.item()] * V * model.y[j.item()]
        II = np.where(Z[:,j.item()]>INT_TOL)
        if DEBUG_CB:
            print([i.item() for i in np.nditer(II)])
        for i in np.nditer(II):
            aij = c[j.item()] * (a_bar[i.item()] + a[i.item()])
            sum_expression +=  aij * model.z[i.item(), j.item()]
    return model.scuts.add(sum_expression <= model.theta)


#@pyomo_callback('my_callback')
def my_callback(cb_m: Any, cb_opt: Any, cb_where: int) -> None:
    r"""Separate an incumbent master solution inside Gurobi's MIP callback."""
    model = cb_m
    mmodel = opt._solver_model

    if cb_where == GRB.Callback.MIPSOL and BRANCH_AND_CUT and model.cuts_added < MAX_CUTS: # and not cb_m.theta.value is None:
        #opt.update()
        #obj = cb_opt.cbGet(GRB.Callback.MIP_OBJBST)
        # MIP solution callback
        nodecnt = mmodel.cbGet(GRB.Callback.MIPSOL_NODCNT)
        obj = mmodel.cbGet(GRB.Callback.MIPSOL_OBJBST)
        theta = mmodel.getVarByName("theta")
        thetaVal = mmodel.cbGetSolution(theta)
        solcnt = mmodel.cbGet(GRB.Callback.MIPSOL_SOLCNT)
        #if model.theta.value is None:
        #    print("No soln found in Pyomo object, callback, sol count: ", solcnt, " obj: ", obj)
        if DEBUG_CB_0:
            print(f"**** New solution at node {nodecnt:.0f}, candidate obj {obj:g}, "f"sol {solcnt:.0f}, theta = {thetaVal:g} ****")
        if DEBUG_CB_2:
            print(mmodel.getVars())
        Z = callback_helper(mmodel,model)
        pp, bb, _ = create_sos_instance(model.a_bar, model.a_hat, model.V, model.c, [], Z)
        p_star, a, constant = convex_pw_knapsack_wrapper(pp, bb, model.Omega, Z, model.a_hat)
        if p_star > thetaVal + INT_TOL:
            cons = add_cut_two(Z, model.a_bar, a, model.V, model.c, model)
            if DEBUG_CB:
                print("found cut, p_star: ", p_star, " theta: ", thetaVal)
                cons.pprint()
            cb_opt.cbLazy(cons)
            model.cuts_added += 1
    #else:
    #raise Exception("invalid callback..")

def print_sol(model: Any) -> None:
    """Display the assignment variables and current master objective."""
    #for var in model.getVars():
    #    val = model.getVal(var)
    #    if val != 0:
    #        print(var,":",model.getVal(var), end=" ")
    pe.display(model.z)
    print("\nmodel obj val: ",pe.value(model.obj.expr))

def create_sos_instance(
    a_bar: np.ndarray,
    a_hat: np.ndarray,
    V: float,
    c: np.ndarray,
    model: Any,
    Z: Any = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Construct the 2PCK breakpoints from a master assignment.

    This is the code realization of Observation 1: each assigned bin becomes
    one function with parameters ``gamma_j``, ``beta_j``, and ``u_j``.
    """
    n = len(a_bar)
    m = len(c)
    #print("n=",n, " m=", m)
    f = {}
    u = {}
    b = np.zeros((m, 3), int)
    p = np.zeros((m, 3), float)
    y = None

    ZZ = None
    if Z is None:
        ZZ = np.zeros([n,m])
    else:
        y = np.amax(Z,axis=0)

    for j in range(m):
        yj = None
        if Z is None:
            yj = model.y[j].value
        #if not Z is None:
        else:
            yj = y[j]
        if  yj > 1 - INT_TOL:
            f[j] = 0
            u[j] = 0
            for i in range(n):
                # print("loop problem")
                zij = None
                if Z is None:
                    zij = model.z[i, j].value
                #if not Z is None:
                else:
                    zij = Z[i,j]
                if  zij > 1 - INT_TOL:
                    if Z is None:
                        ZZ[i, j] = 1
                    f[j] += a_bar[i]
                    u[j] += a_hat[i]
            b[j, 1] = min(max(V-f[j],0), min(BOUND_OVERFILL, u[j]))  # for mid breakpoint - min of unfilled capacity and u, positive part
            b[j, 2] = min(BOUND_OVERFILL, u[j])  ## testing with bounds on u which applies with equal c's           # ,axis=0) #max(u[j] - V + f[j], 0)]]), axis=0)
            p[j, 0] = c[j] * max(f[j] - V, 0)
            p[j, 1] = c[j] * max(f[j] - V, 0)  # 0 if started neg
            # p[j,2] = c[j] * max(u[j] - V + f[j], 0)
            p[j, 2] = c[j] * (b[j, 2] - b[j, 1] + max(f[j] - V, 0))
            if abs(p[j, 2]) <= EPS:  # if b[j,1]=u[j] < f[j] - V
                b[j, 1] = 0
                b[j, 2] = 0
            assert p[j, 2] >= p[j, 1]
            assert p[j, 1] >= p[j, 0]
    return p,b, ZZ

def solve_instance(
    a_bar: np.ndarray,
    a_hat: np.ndarray,
    V: float,
    c: np.ndarray,
    Omega: float,
    timelimit: float = TIME_LIMIT,
) -> Tuple[Any, bool, float, float, float, int, float, float, int]:
    r"""Run Algorithm 1 until no violated scenario remains.

    Returns assignment and timing statistics used by the case-study and
    benchmark scripts.
    The master variables ``(Y^*, Z^*, Theta^*)`` are solved, then the inner
    maximization returns ``eta^*``. A new scenario is generated whenever
    ``eta^* > Theta^* + VIOL_TOL``.
    """
    n = len(a_bar)
    m = len(c) #int(math.ceil(2 * (sum(a_bar) + Omega) / V))
    scenario_num = 0
    #global model
    model = rebppinit_pyomo(a_bar, a_hat, V, c,Omega)
    #model.Omega = Omega

    print("solving.. Tau0=", TAU0, " Tau1=", TAU1, " Cuts: ", BRANCH_AND_CUT)
    global opt
    opt.set_instance(model,symbolic_solver_labels=True)

    if BRANCH_AND_CUT or VALIDINEQ2:
        #opt.set_gurobi_param('PreCrush', 1)
        #opt.set_gurobi_param('LazyConstraints', 1)
        #opt.options['PreCrush']=1
        #opt.options['VarBranch'] = 2
        opt.options['LazyConstraints'] = 1
        #opt.options['PreCrush'] = 1
        opt.options['Presolve'] = 2
        opt.options['DisplayInterval'] = 5000
        opt.set_callback(my_callback)

    gapVal = TAU0
    start = time.time()

    it = 0
    numBins = 0
    masterTime = 0
    timeL = False
    p_star_old = math.inf
    a_old = []
    z_old = []
    #model_old = []
    while True:
        opt.set_instance(model)
        opt.options["MIPGap"] = gapVal
        opt.options['TimeLimit'] = int(float(timelimit)-(time.time()-start))
        masterStart = time.time()
        #opt.update()
        #gModel = opt._solver_model
        #gModel.setParam('OutputFlag', 0)
        # Set up callback function with required arguments
        #callback_data = CallbackData(gModel.getVars())
        #callback_func = partial(my_callback_g, cbdata=callback_data)
        #gModel.optimize(callback_func)
        solverOutput = False
        if DEBUG_CB or DEBUG_INEQ_NOVAR or GUROBI_OUTPUT:
            solverOutput = True
        if it > 0 and MIP_START_OR_HINT:
            for k in z_old:
                if MIP_START_OR_HINT == 2:
                    opt.set_var_attr(model.z[k], 'VarHintVal', 1)
                else:
                    opt.set_var_attr(model.z[k], 'Start', 1)

        results = opt.solve(model, tee=solverOutput)
        masterTime += time.time() - masterStart
        soln = results.Solution
        status = results.Solver.status  # results.Solver()['Termination condition'].value
        if results.solver.termination_condition == TerminationCondition.maxTimeLimit:
            print('time limit!')
            timeL = True
            break
        elif status != SolverStatus.ok:  # TerminationCondition.optimal: #'optimal':
            OSError('error occurred, status: {}.  Check model!'.format(status))

        p,b,Z = create_sos_instance(a_bar,a_hat,V,c,model)
            # p = np.append(p, np.array([[c[j] * max(f[j] - V, 0), c[j] * max(f[j] - V, 0), c[j] * max(u[j] - V + f[j], 0)]]), axis=0)
        if __DEBUG:
            print("Before running convex knapsack, Omega=", Omega, " p=", p, " b=", b)
        # if np.sum(p[:,2]) > NZ_TOl:
        # p_star_0, a_0 = convex_pw_knapsack_wrapper(p, b, Omega, model.z.extract_values(), a_hat, model, True)

        p_star, a, constant = convex_pw_knapsack_wrapper(p, b, Omega, Z, a_hat, SOS_SOLVE)  # False)  # if last argument is false then DP is invoked otherwise SoS
        p_star_0 = p_star

        theta_star = model.theta.value
        print("iteration: ", it, " p_star val: ", p_star, " theta_star: ", theta_star, " constant=", constant, " obj=", pe.value(model.obj.expr), " cuts: ", model.cuts_added, "******")
        if DEBUG_INEQ_NOVAR:
            print_sol(model)

        if False: #(p_star == p_star_old and a == a_old and p_star > theta_star + 10*VIOL_TOL):
            print("got same subprob p_star=", p_star, " p_star_old", p_star_old, p_star_0)
            print(a)
            print(a_old)
            print(p)
            print(b)
            print(Omega)
            for k in model.z.keys():
                if abs(model.z[k].value) > 1e-2:
                    print(model.z[k].getname(), model.z[k].value, end=' ')
            print('***')
            print(z_old)
            model_old.scuts.pprint()
            #model, alpha = update_rebpp_pyomo(model, a_bar, V, c, a, scenario_num)   ## for debugging
            print('***')
            model.scuts.pprint()
            raise Exception("breaking..")

        p_star_old = p_star
        a_old = a
        model_old = deepcopy(model)
        z_old = []
        for k in model.z.keys():
            if abs(model.z[k].value) > 1e-2:
                z_old.append(k)
                #z_old.append(model.z[k].getname())
        it += 1

        rTime = time.time() - start
        if rTime >= TIME_LIMIT:
            print("time limit")
            break
        elif p_star <= theta_star + VIOL_TOL or (BRANCH_AND_CUT and MAX_CUTS == math.inf):
            if gapVal == TAU1: #or BRANCH_AND_CUT:
                print("terminating, could not find a constraint violating by more than tol=", VIOL_TOL)
                # print_sol(model)
                numBins = sum(model.y[j].value for j in model.J)
                if DEBUG_CB_2:
                    model.display()
                break
            else:
                gapVal = TAU1
                print("setting gapVal: ", gapVal)
        model, alpha = update_rebpp_pyomo(model, a_bar, V, c, a, scenario_num)
        opt.update()
        scenario_num += 1
    runTime = time.time() - start
    print(" elapsed time: ", runTime)
    assign = pd.DataFrame(columns=["patient", "shift"])
    for k in model.z.keys():
        if abs(model.z[k].value) > 1e-2:
            assign = pd.concat([assign,pd.DataFrame(np.array([[k[0], k[1]]]), columns=assign.columns)],ignore_index=True)
    return assign, timeL, runTime, masterTime, numBins, scenario_num, model.theta.value, pe.value(model.obj.expr), model.cuts_added

if __name__ == "__main__":
    # example problem

    Omega = Omega_val # also B
    # 4
    MAX_BINS = 5 #20
    MAX_SCENRIOS = 1e3
    # 2
    OT_cost = 4/BINSIZE #0.6  # 1.3/BINSIZE #0.0021 #0.003
    # 0.6
    #

    pe.ConcreteModel.getVal = classmethod(getVal)

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
    #c += EPS*np.array(range(m))

    # assign, timeL, runTime, masterTime, numBins, scenario_num, model.theta.value, pe.value(model.obj.expr), model.cuts_added
    assign,timeL, runTime, masterTime, numBins, scenario_num, thetastar, obj, cuts_added = solve_instance(a_bar, a_hat, V, c,Omega)
    print(" time limit: ", timeL, "run time: ", runTime, " master runtime: ", masterTime, " num of bins: ", numBins, " theta*: ", thetastar, " obj=", obj, " num of scenarios: ", scenario_num, " cuts added:", cuts_added)
    #error("quit")
    print(assign)
    assign.to_csv(FILENAME + "_schedule_" + str(V) + "_" + str(Omega) + ".csv")
