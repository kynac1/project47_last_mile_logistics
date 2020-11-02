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


def find_opt_collection(
    k, CUSTOMERS, FACILITY, fac_lat, fac_lon, dist, weight, demand, Fac_cap
):
    """
    k: number of collection points
    CUSTOMERS: set of customers by suburb
    FACILITY: set of potential collection points
    fac_lat: latitudes for the potential collection points
    fac_lon: longitudes for the potential collection points
    dist: distance matrix from the customer to the collection point
    weight: the weighting of suburb
    demand: the number of packages that suburb demands to be sent to collection points
    Fac_cap: capacity of each collection point

    This function models the collection location optimisation problem using PuLp
    and solves the model using CBC solver.
    """
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
    )

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
    TOL = 0.00001

    for j in FACILITY:
        if use_vars[j].varValue > TOL:
            sol_fac_lat.append(fac_lat[j])
            sol_fac_lon.append(fac_lon[j])
            print("Establish facility at site ", j)

    # for v in prob.variables():
    # print(v.name, " = ", v.varValue)

    print("The cost of travel = ", value(prob.objective))

    return sol_fac_lat, sol_fac_lon
