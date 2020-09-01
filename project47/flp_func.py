from pulp import *
import numpy as np
from project47.routing import *
from project47.customer import Customer
from project47.data import get_sample, read_data
from project47.multiday_simulation import *
from project47.flp_data import *
# from project47.flp_data import *
from numpy.random import Generator, PCG64
import os
import matplotlib.pyplot as plt

# CUSTOMERS = [1,2,3,4,5]
# FACILITY = ['f1','f2','f3']
# Fac_cost = {'f1': 5,
#             'f2': 10,
#             'f3': 10}

# Fac_cap = {'f1': 50,
#             'f2': 50,
#             'f3': 50}

# dist = {'f1': {1: 4, 2: 5, 3: 6, 4: 8, 5: 10},
#         'f2': {1: 6, 2: 4, 3: 3, 4: 5, 5: 8},
#         'f3': {1: 9, 2: 7, 3: 4, 4: 3, 5: 4}}

# k = 10

# set problem
prob = LpProblem("FacilityLocation", LpMinimize)

# desicion variables
# if facility j serves customer i
serv_vars = LpVariable.dicts("x",
                                 [(i, j) for i in CUSTOMERS
                                         for j in FACILITY],
                                 cat=LpBinary)
# if facility j is used
use_vars = LpVariable.dicts("y", FACILITY, cat=LpBinary)

# objective function
prob += lpSum(dist[j][i] * serv_vars[i, j] for j in FACILITY for i in CUSTOMERS) #+ lpSum(Fac_cost[j] * use_vars[j] for j in FACILITY), "min_dist"

# constraints
# each package should be delivered to a facility
for i in CUSTOMERS:
    prob += lpSum(serv_vars[(i, j)] for j in FACILITY) == 1

# capacity constraint
for j in FACILITY:
    prob += lpSum(serv_vars[(i, j)] for i in CUSTOMERS) <= Fac_cap[j] * use_vars[j]

# number of collection points
    prob += lpSum(use_vars[j] for j in FACILITY) == k


# upper bound for x, tight formlation
for i in CUSTOMERS:
    for j in FACILITY:
        prob += serv_vars[(i, j)] <= use_vars[j]

# solution
prob.solve()
print("Status: ", LpStatus[prob.status])

TOL = .00001
for j in FACILITY:
    if use_vars[j].varValue > TOL:
        print("Establish facility at site ", j)

# for v in prob.variables():
#     print(v.name, ' = ', v.varValue)

print("The cost of production in dollars for one year= ", value(prob.objective))