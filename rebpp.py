from pyscipopt import Model, quicksum, SCIP_PARAMSETTING
from knapsack import rebppinit, update_rebpp
from sos2 import sos2, convex_pw_knapsack_dp
import numpy as np

a_hat = [2,2,2,2]
a_bar = [2,2,3,1]
Omega = 3 # also B
V = 8
VIOL_TOL = 1e-6
#first fit decreasing heuristic for robust bin packing with omega-uncertainty

def FFD_RBPP(a_bar, a_hat, V, Omega):
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
    n = len(a_hat)
    m = len(p)
    a = [0] * n

    t_val1 = [0] * m
    t_val2 = [0] * m
    p_star = None
    items = None
    i_max = None

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
                    # print("a[i]: ",a[i])
                    # print("a_hat: ",a_hat)
                    # print("a_hat_totla: ", a_hat_total)
                    a[i] = min(a_hat[i],a_hat_total) #this is a_hat[i] correct?
                    a_hat_total -= a[i]

    else:
        p_star, items, i_max = convex_pw_knapsack_dp(p,b,Omega)
        print("items:", items)
        fullDevSum = 0
        for item in items:
            print("made it")
            print("item: ",item)
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
    for var in model.getVars():
        print(var,":",model.getVal(var))
    print("model obj val: ",model.getObjVal())

print(FFD_RBPP(a_bar,a_hat,V, Omega))
if __name__ == "__main__":
    
    a_bar = [2,2,3,1]
    a_hat = [2,2,2,2]
    Omega = 3 # also B
    V = 3
    c = [2,2]
    m = len(c)
    n = len(a_bar)
    alpha = {}
    scenario_num = 0
    model, theta, y, f_bar, z = rebppinit(a_bar,a_hat,V,c)
    model.hideOutput()
    # variables_list = model.getVars()
    # for var in variables_list:
    #     print(f"Variable {var.name}")

    # theta,y,f_bar,z = model.getVars()

    while True:
        f = {}
        u = {}
        # model2 = Model(sourceModel = model) # Linear relxation of RMP
 
        # model.setPresolve(SCIP_PARAMSETTING.OFF)
        # model.setHeuristics(SCIP_PARAMSETTING.OFF)
        # model.disablePropagation()
        model.optimize()
        print("master")
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
            print(p)
            print(len(p))
            print(n)
            break


        model,alpha = update_rebpp(model, a_bar, V, c, a, theta, y, z, alpha, scenario_num)
        scenario_num += 1
        # model.writeLP("after_update_model.lp")
                    
        # def update_rebpp(model, a_bar, V, c, a, theta, y, f_bar,z):
