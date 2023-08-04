from pyscipopt import Model, quicksum, SCIP_PARAMSETTING
from knapsack import rebppinit, update_rebpp
from sos2 import sos2, convex_pw_knapsack_dp
import numpy as np
import csv
import pandas as pd


"""
sample small test case
"""
a_hat = [2,2,2,2]
a_bar = [2,2,3,1]
Omega = 3 # also B
V = 8
VIOL_TOL = 1e-6

#first fit decreasing heuristic for robust bin packing with omega-uncertainty

def FFD_RBPP(a_bar, a_hat, V, Omega):
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
            if nominal_fill[j] + min(Omega, deviation_fill[j] + a_hat[index]) + a_bar[index] <= V:
                nominal_fill[j] += a_bar[index]
                deviation_fill[j] += min(a_hat[index], Omega - deviation_fill[j])
                sol[j].append(index)
                break
            else:
                sol.append([index])
                nominal_fill.append(a_bar[index])
                deviation_fill.append(min(a_hat[index], Omega))
    return sol

def convex_pw_knapsack(p, b, Omega, z, a_hat, model, sos = True):
    """
    used to retrive the p_star and a values
    """
    n = len(a_hat)
    m = len(p)
    a = [0] * n

    t_val1 = [0] * m
    t_val2 = [0] * m
    p_star = None
    items = None
    i_max = None

    # if we use the scip version
    if sos:
        p_star,knapsack_model,t = sos2(p,b,Omega)
        for j in range(m):
            t_val1 = knapsack_model.getVal(t[j,1])
            t_val2 = knapsack_model.getVal(t[j,2])
            print("t_values:",t_val1,t_val2)
        if t_val2 == 1:
            for i in range(n):
                if model.getVal(z[i,j]) == 1:
                    a[i] = a_hat[i]
        elif t_val1 > 0:
            a_hat_total = b[j,1] * t_val1 + b[j,2] * t_val2
            for i in range(n):
                if model.getVal(z[i,j]) == 1 and a_hat_total > 0:
                    a[i] = min(a_hat[i],a_hat_total)
                    a_hat_total -= a[i]
    # if we use the dp method
    else:
        p_star, items, i_max = convex_pw_knapsack_dp(p,b,Omega)
        fullDevSum = 0
        for item in items:
            for i in range(n):
                if model.getVal(z[i,item]) == 1:
                    a[i] = a_hat[i]
                    fullDevSum += a_hat[i]
        remDev = Omega - fullDevSum
        for i in range(n):
            if model.getVal(z[i,i_max]) == 1:
                a[i] = min(a_hat[i],remDev)
                remDev -= a[i]

    return p_star, a

def print_sol(model):
    """
    used to print the values of the variables and the objective value
    """
    for var in model.getVars():
        print(var,":",model.getVal(var))
    print("model obj val: ",model.getObjVal())

if __name__ == "__main__":
    # example problem

    a_hat = []
    Omega = 240 # also B
    V = 480
    c = [0.005,0.005,0.005,0.005,0.005]
    BEGIN = 0
    END = 50

    rambam_data = pd.read_csv("Dep13300with_a_ahat.csv")
    a_bar = rambam_data["a"]
    a_hat = rambam_data["ahat"]
    a_bar = np.round(a_bar[BEGIN:END].to_numpy())
    a_hat = np.round(a_hat[BEGIN:END].to_numpy())
    a_bar = np.asarray(a_bar, dtype = 'int')
    a_hat = np.asarray(a_hat, dtype = 'int')




    # with open(".csv", 'r') as file:
    #     csvreader = csv.reader(file)
    #     for row in csvreader:
    #         a_bar.append(int(row[-2]))
    #         a_hat.append(int(row[-1]))
    #     a_bar = a_bar[1:]
    #     a_hat = a_hat[1:]
            

    m = len(c)
    n = len(a_bar)

    
    alpha = {}
    scenario_num = 0
    model, theta, y, f_bar, z = rebppinit(a_bar,a_hat,V,c)
    model.hideOutput()

    while True:
        f = {}
        u = {}

        model.optimize()
        print_sol(model)
        
        model.writeProblem("model" + str(iter) + ".cip",trans=False)
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
                if model.getVal(z[i,j]) == 1:
                    f[j] += a_bar[i]
                    u[j] += a_hat[i]

            b = np.append(b, np.array([[0, max(V-f[j],0), max(u[j],0)]]), axis=0)
            p = np.append(p, np.array([[0, c[j]*max(f[j]-V,0), c[j]*max(u[j]-V+f[j],0)]]), axis=0)
            
        theta_star = model.getVal(theta)
        
        p_star,a = convex_pw_knapsack(p,b,Omega,z,a_hat,model,False)

        print("p_star val, theta_star_val:",p_star,theta_star)

        if p_star <= theta_star + VIOL_TOL:
            print_sol(model)
            break


        model,alpha = update_rebpp(model, a_bar, V, c, a, theta, y, z, alpha, scenario_num)
        scenario_num += 1
        # model.writeLP("after_update_model.lp")
                    
        # def update_rebpp(model, a_bar, V, c, a, theta, y, f_bar,z):
