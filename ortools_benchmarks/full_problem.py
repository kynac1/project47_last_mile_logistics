import numpy as np
from ortools.constraint_solver import routing_enums_pb2
import os
from project47.data import *
from project47.routing import *
from project47.customer import Customer
from project47.multiday_simulation import multiday
from project47.simulation import *
from functools import reduce
from collections import defaultdict

from time import time

import json


data = defaultdict(lambda: [])


def dist_and_time(customers):
    return osrm_get_dist(
        "",
        "",
        [customer.lat for customer in customers],
        [customer.lon for customer in customers],
        host="localhost:5000",
        save=False,
    )


def simulator(
    routes, dm, tm, delivery_time_windows, customers, rg: np.random.Generator
):
    return sim(routes, wait_policy(dm, tm, delivery_time_windows, customers, rg))


def route_optimizer(
    depots,
    dm,
    tm,
    time_windows,
    day,
    arrival_days,
    futile_count,
    alternate_locations,
    fss=routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
    lsm=routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT,
    tlim=100,
):
    locs = dm.shape[0]
    r = ORToolsRouting(locs, 5)
    dim, ind = r.add_dimension(dm, 0, 50000, True, "distance")
    r.routing.SetArcCostEvaluatorOfAllVehicles(ind)
    dim, ind = r.add_time_windows(tm, time_windows, 28800, 28800, False, "time")
    for alternates in alternate_locations:
        r.add_option(alternates, 50000)

    r.search_parameters.first_solution_strategy = fss

    r.search_parameters.local_search_metaheuristic = lsm
    s = r.solve(tlim=tlim, log=True)

    fss_indexer = {
        v: k for k, v in routing_enums_pb2.FirstSolutionStrategy.__dict__.items()
    }
    lsm_indexer = {
        v: k for k, v in routing_enums_pb2.LocalSearchMetaheuristic.__dict__.items()
    }
    print(tlim, " : ", r.objective)
    global data
    data[str((fss_indexer[fss], lsm_indexer[lsm], tlim))].append(r.objective)

    unscheduled = []
    scheduled = reduce(lambda x, y: x + y, s.routes)
    for i in range(locs):
        if i not in scheduled:
            unscheduled.append(i)
    return s, unscheduled


def benchmarker(fss, lsm, tlim):
    """This is the main example of all the functionality.

    The idea is that when we create a new experiment to run, we'd copy the structure of this function and replace
    parts so that it implements the new policies
    """
    sample_generator = test_sample_generator()
    start = time()
    data = multiday(
        np.array([[-43.5111688], [172.7319266]]),
        sample_generator,
        dist_and_time,
        lambda *args: route_optimizer(*args, fss=fss, lsm=lsm, tlim=tlim),
        simulator,
        1,
        0,
        28800,
        seed=123456789,
        replications=1,
    )
    data = multiday(
        np.array([[-43.5111688], [172.7319266]]),
        sample_generator,
        dist_and_time,
        lambda *args: route_optimizer(*args, fss=fss, lsm=lsm, tlim=tlim),
        simulator,
        1,
        0,
        28800,
        seed=532409213,
        replications=1,
    )
    data = multiday(
        np.array([[-43.5111688], [172.7319266]]),
        sample_generator,
        dist_and_time,
        lambda *args: route_optimizer(*args, fss=fss, lsm=lsm, tlim=tlim),
        simulator,
        1,
        0,
        28800,
        seed=98743457,
        replications=1,
    )

    return time() - start


def test_sample_generator():
    cd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    sample_data = os.path.join(cd, "Toll_CHC_November_Sample_Data.csv")
    CHC_data = os.path.join(cd, "christchurch_street.csv")
    sample_df, CHC_df, _, CHC_sub, CHC_sub_dict = read_data(
        sample_data,
        CHC_data,
        lat_min=-43.6147000,
        lat_max=-43.4375000,
        lon_min=172.4768000,
        lon_max=172.7816000,
    )

    def sample_generator(rg: np.random.Generator):
        lat, lon = get_sample(
            100, rg, cd, sample_df, CHC_df, CHC_sub, CHC_sub_dict, save=False
        )
        time_windows = np.zeros((len(lat), 2))
        for i in range(len(lat)):
            if rg.random() > 0.5:
                time_windows[i, 0] = 0
                time_windows[i, 1] = 14400
            else:
                time_windows[i, 0] = 14400
                time_windows[i, 1] = 28800

        customers = [Customer(lat[i], lon[i], 0.8, 0.8, rg=rg) for i in range(len(lat))]

        return customers, time_windows

    return sample_generator


import sys

if __name__ == "__main__":
    # routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC,
    fss_list = [
        routing_enums_pb2.FirstSolutionStrategy.ALL_UNPERFORMED,
        routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES,
        routing_enums_pb2.FirstSolutionStrategy.FIRST_UNBOUND_MIN_VALUE,
        routing_enums_pb2.FirstSolutionStrategy.GLOBAL_CHEAPEST_ARC,
        routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_ARC,
        routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION,
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
        routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC,
        routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
        routing_enums_pb2.FirstSolutionStrategy.SEQUENTIAL_CHEAPEST_INSERTION,
    ]
    lsm_list = [
        routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC,
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
        routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT,
        routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
    ]

    # benchmarker(
    #    [100],
    #    fss_list,
    #    [routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT],
    # )
    fss_indexer = {
        v: k for k, v in routing_enums_pb2.FirstSolutionStrategy.__dict__.items()
    }
    lsm_indexer = {
        v: k for k, v in routing_enums_pb2.LocalSearchMetaheuristic.__dict__.items()
    }

    for fss in fss_list:
        for lsm in lsm_list:
            print("FSS: ", fss_indexer[fss], file=sys.stderr)
            print("LSM: ", lsm_indexer[lsm], file=sys.stderr)
            sys.stderr.flush()
            for tlim in [30]:
                try:
                    benchmarker(
                        fss,
                        lsm,
                        tlim,
                    )
                except:
                    print("FAILED")

                # with open("ortools_benchmarks/current_results4.json", "w") as f:
                #    json.dump(data, f)

# Run with python full_problem.py 2> full_problem.log to record results