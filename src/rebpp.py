from pyscipopt import Model, quicksum, SCIP_PARAMSETTING
#from knapsack import rebppinit, update_rebpp
from sos2 import sos2, convex_pw_knapsack_dp
import numpy as np
import pandas as pd
import pyomo.environ as pe


"""
sample small test case
"""
__DEBUG_2 = False

a_hat = [2,2,2,2]
a_bar = [2,2,3,1]
Omega = 3 # also B
V = 8
VIOL_TOL = 1e-6
INT_TOL = 1e-3

MAX_SCENRIOS = 1e4


def getVal(model, var):
    return var

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

    mdl.objCons = pe.Constraint(expr = sum(c[j]*mdl.alpha_bar[j] for j in mdl.J)<= mdl.theta)

    mdl.obj = pe.Objective(expr = sum(mdl.y[j] for j in mdl.J) + mdl.theta, sense=pe.minimize)

    mdl.scuts = pe.ConstraintList()

    return mdl, mdl.theta, mdl.y, mdl.alpha_bar, mdl.z



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


def update_rebpp_pyomo(mdl, a_bar, V, c, a, theta, y, z, alpha, scenario_num):
    """
    used to recieve new model and alph
    """
    if __DEBUG_2:
        print("scenario_num: ", scenario_num)

    #mdl.scenarios = mdl.scenarios | pe.Set(initialize=[scenario_num])
    mdl.sn = scenario_num
    for j in mdl.J:
        #mdl.alpha_bar[j, scenario_num] #= mdl.add_column(mdl,0,[],[])
        mdl.scuts.add(sum(mdl.z[i, j] * (a_bar[i] + a[i]) for i in mdl.I) <= V * mdl.y[j] + mdl.alpha[j, scenario_num])

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
    #t_val1 = [0] * m
    #t_val2 = [0] * m
    p_star = None
    items = []
    i_max = None
    # if we use the scip version
    if sos:
        if __DEBUG_2:
            p_star, items, i_max = convex_pw_knapsack_dp(p, b, Omega, True)  # true
            print("p_star knapsack = ", p_star, " items=", items, " i_max=", i_max)
        p_star, items, i_max, _, _ = sos2(p, b, Omega)
        if __DEBUG_2:
            print("p_star sos = ", p_star, " items=", items, " i_max=", i_max)
    else:
        p_star, items, i_max = convex_pw_knapsack_dp(p,b,Omega,True) # true since y intercept is nonzero
    fullDevSum = 0
    for item in items:
        for i in range(n):
            if model.getVal(z[i,item]) == 1:
                a[i] = a_hat[i]
                fullDevSum += a_hat[i]
    remDev = Omega - fullDevSum
    if i_max is not None:
        for i in range(n):
            if model.getVal(z[i,i_max]) == 1:
                a[i] = min(a_hat[i],remDev)
                remDev -= a[i]
    if p_star > 0 and remDev > 0:
        print ("items=",items, " i_max=", i_max, " remDev=", remDev, " p=", p, " b=", b)
        raise ValueError("remDev>0")
    return p_star, a

def print_sol(model):
    """
    used to print the values of the variables and the objective value
    """
    for var in model.getVars():
        val = model.getVal(var)
        if val != 0:
            print(var,":",model.getVal(var), end=" ")
    print("\nmodel obj val: ",model.getObjVal())

if __name__ == "__main__":
    # example problem

    a_hat = []
    Omega = 240 # also B
    V = 480
    c = [3e-3]*8 #[0.003,0.003,0.003,0.003,0.003,0.003,0.003,0.003]
    BEGIN = 0
    END = 50

    rambam_data = pd.read_csv("../data/Dep13300with_a_ahat.csv")
    a_bar = rambam_data["a"]
    a_hat = rambam_data["ahat"]
    a_bar = np.round(a_bar[BEGIN:END].to_numpy())
    a_hat = np.round(a_hat[BEGIN:END].to_numpy())
    a_bar = np.asarray(a_bar, dtype = 'int')
    a_hat = np.asarray(a_hat, dtype = 'int')

    m = len(c)
    n = len(a_bar)
    alpha = {}
    scenario_num = 0
    model, theta, y, f_bar, z = rebppinit(a_bar,a_hat,V,c)
    model.hideOutput()
    it = 0

    while True:
        f = {}
        u = {}
        model.optimize()
        #print_sol(model)
        #model.writeProblem("model" + str(iter) + ".cip",trans=False)
        # model = model2
        b = np.empty((0,3), int)
        p = np.empty((0,3), int)
        if model.getStatus() != "optimal":
            print("Error (suboptimal)")
            raise ValueError
        for j in range(m):
            f[j] = 0
            u[j] = 0
            for i in range(n):
                # print("loop problem")
                if model.getVal(z[i,j]) > 1 - INT_TOL:
                    f[j] += a_bar[i]
                    u[j] += a_hat[i]
            b = np.append(b, np.array([[0, max(V-f[j],0), max(u[j]-V+f[j],0)]]), axis=0)
            p = np.append(p, np.array([[c[j]*max(f[j]-V,0), c[j]*max(f[j]-V,0), c[j]*max(u[j]-V+f[j],0)]]), axis=0)

        print(b)
        print(p)
        theta_star = model.getVal(theta)
        
        p_star,a = convex_pw_knapsack_wrapper(p,b,Omega,z,a_hat,model,False)
        it += 1
        print("iteration: ", it, " p_star val: ", p_star, " theta_star_val: ",theta_star)

        if p_star <= theta_star + VIOL_TOL:
            print_sol(model)
            break
        model,alpha = update_rebpp(model, a_bar, V, c, a, theta, y, z, alpha, scenario_num)
        scenario_num += 1
        # model.writeLP("after_update_model.lp")
        # def update_rebpp(model, a_bar, V, c, a, theta, y, f_bar,z):
