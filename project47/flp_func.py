from pulp import *
import numpy as np
from project47.routing import *
from project47.customer import Customer
from project47.data import get_sample, read_data

# from project47.multiday_simulation import *
# from project47.flp_data import *
# from project47.flp_data import *
from numpy.random import Generator, PCG64
import os
import matplotlib.pyplot as plt

# from mpl_toolkits.basemap import Basemap
import matplotlib.pylab as pylab
import string
import matplotlib.cm as cm
import pandas as pd

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


def find_opt_collection(
    k, CUSTOMERS, FACILITY, lat, lon, fac_lat, fac_lon, dist, weight, demand, Fac_cap
):
    # set problem
    prob = LpProblem("FacilityLocation", LpMinimize)

    # desicion variables
    # if facility j serves customer i
    serv_vars = LpVariable.dicts("x", [(i, j) for i in CUSTOMERS for j in FACILITY], 0)
    # if facility j is used
    use_vars = LpVariable.dicts("y", FACILITY, cat=LpBinary)

    # objective function
    prob += lpSum(
        dist[j][i] * serv_vars[i, j] * weight[i] for j in FACILITY for i in CUSTOMERS
    )  # + lpSum(Fac_cost[j] * use_vars[j] for j in FACILITY), "min_dist"
    # prob += lpSum(dist[j][i] * serv_vars[i, j] for j in FACILITY for i in CUSTOMERS) #+ lpSum(Fac_cost[j] * use_vars[j] for j in FACILITY), "min_dist"

    # constraints
    # each package should be delivered to a facility
    for i in CUSTOMERS:
        prob += lpSum(serv_vars[(i, j)] for j in FACILITY) >= demand[i]

    # # capacity constraint
    # for j in FACILITY:
    #     prob += lpSum(serv_vars[(i, j)] for i in CUSTOMERS) <= Fac_cap[j] * use_vars[j]

    # number of collection points
    prob += lpSum(use_vars[j] for j in FACILITY) == k

    # upper bound for x, tight formulation
    for i in CUSTOMERS:
        for j in FACILITY:
            prob += serv_vars[(i, j)] <= use_vars[j] * demand[i]

    # solution
    prob.solve()
    print("Status: ", LpStatus[prob.status])

    sol_fac_lat = []
    sol_fac_lon = []
    assigned_lat = []
    assigned_lon = []
    TOL = 0.00001

    for j in FACILITY:
        if use_vars[j].varValue > TOL:
            sol_fac_lat.append(fac_lat[j])
            sol_fac_lon.append(fac_lon[j])
            print("Establish facility at site ", j)
            l_lat = []
            l_lon = []
            for i in CUSTOMERS:
                if serv_vars[(i,j)].varValue > TOL:
                    l_lat.append(lat[i])
                    l_lon.append(lon[i])
            assigned_lat.append(l_lat)
            assigned_lon.append(l_lon)

    for i in CUSTOMERS:
        for j in FACILITY:
            print(serv_vars[(i,j)].varValue)

    # for v in prob.variables():
    # print(v.name, " = ", v.varValue)

    print("The cost of travel = ", value(prob.objective))

    # m = Basemap(llcrnrlon=172.4768000,llcrnrlat=-43.6147000,urcrnrlon=172.7816000,urcrnrlat=-43.4375000,lat_ts=20,
    #             resolution='h',projection='merc',lon_0=172.4768000,lat_0=-43.6147000)
    # lat1, lon1 = m(lat, lon)
    # m.drawmapboundary(fill_color='white') # fill to edge
    # m.scatter(lat1, lon1 ,s=5,c='r',marker="o",cmap=cm.jet,alpha=1.0)

    # sol_fac_coord = list(map(list, zip(sol_fac_lat, sol_fac_lon)))

    return sol_fac_lat, sol_fac_lon, assigned_lat, assigned_lon