import os
from project47.customer import markov_presence
import numpy as np
from project47.simulation import *
from project47.data import *
from project47.routing import *
from project47.multiday_simulation import *
from functools import reduce
from multiprocessing import Process, Pool

import logging
import multiprocessing_logging

multiprocessing_logging.install_mp_handler()


logging.basicConfig(
    filename="experiments/multiaddress_results2/log.txt",
    filemode="w",
    level="DEBUG",
    format="%(processName)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def multiaddress(
    arrival_rate,
    num_vehicles,
    num_time_windows,
    num_addresses,
    policy,
):
    logger.debug("Starting Experiment:")
    logger.debug(
        "Arrival rate = %s, No. Vehicles = %s, No. Time Windows = %s, Addresses per Customer = %s, Policy = %s"
        % (arrival_rate, num_vehicles, num_time_windows, num_addresses, policy.__name__)
    )

    day_start = 0
    day_end = 28800

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
            arrival_rate * num_addresses,  # rg.poisson(arrival_rate) * num_addresses,
            rg,
            cd,
            sample_df,
            CHC_df,
            CHC_sub,
            CHC_sub_dict,
            save=False,
        )
        customers = [
            Customer(
                lat[i],
                lon[i],
                0.9,
                0.9,
                rg=rg,
                presence_interval=28800 // 16,
                presence=markov_presence(16, 0.2, rg),
            )
            for i in range(len(lat))
        ]
        num_actual_packages = len(customers) // num_addresses
        for i in range(len(customers)):
            customers[i].add_alternate(customers[i % num_actual_packages])

        time_windows = np.zeros((len(customers), 2))
        for i in range(len(customers)):
            time_windows[i, :] = customers[i].get_time_window(
                [
                    [i, i + 28800 // num_time_windows]
                    for i in range(0, 28800, 28800 // num_time_windows)
                ]
            )
        """for i in range(len(lat)):
            interval = (day_end - day_start) / num_time_windows
            for j in range(num_time_windows):
                if rg.random() > (num_time_windows - (j + 1)) / num_time_windows:
                    time_windows[i, 0] = interval * j
                    time_windows[i, 1] = interval * (j + 1)
                    break"""

        return customers, time_windows

    def dist_and_time(customers):
        return osrm_get_dist(
            "",
            "",
            [customer.lat for customer in customers],
            [customer.lon for customer in customers],
            host="0.0.0.0:5000",
            save=False,
        )

    def route_optimizer(
        depots,
        dm,
        tm,
        time_windows,
        day,
        arrival_days,
        futile_count,
        alternate_locations,
        fss=routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC,
        lsm=routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        tlim=30,
    ):
        locs = dm.shape[0]
        r = ORToolsRouting(locs, num_vehicles)
        dim, ind = r.add_dimension(dm, 0, 1000000, True, "distance")
        r.routing.SetArcCostEvaluatorOfAllVehicles(ind)
        dim.SetGlobalSpanCostCoefficient(100)
        dim, ind = r.add_time_windows(tm, time_windows, day_end, day_end, False, "time")
        for alternates in alternate_locations:
            r.add_option(alternates, 5000000)

        r.search_parameters.first_solution_strategy = fss

        r.search_parameters.local_search_metaheuristic = lsm
        # r.search_parameters.use_cp_sat = True
        s = r.solve(tlim=tlim, log=True)

        unscheduled = []
        scheduled = reduce(lambda x, y: x + y, s.routes)
        for i in range(locs):
            if i not in scheduled:
                unscheduled.append(i)
        return s, unscheduled

    def simulator(
        routes, dm, tm, delivery_time_windows, customers, rg: np.random.Generator
    ):
        return sim(routes, policy(dm, tm, delivery_time_windows.copy(), customers, rg))

    data = multiday(
        np.array([[-43.5111688], [172.7319266]]),
        sample_generator,
        dist_and_time,
        route_optimizer,
        simulator,
        5,
        day_start,
        day_end,
        plot=False,
        seed=2123897,
    )
    fname = f"experiments/multiaddress_results2/constant_{arrival_rate}_{num_vehicles}_{num_time_windows}_{num_addresses}_{policy.__name__}.json"
    with open(
        fname,
        "x",
    ) as outfile:
        json.dump(data, outfile, separators=(",", ":"))

    logger.debug("Finished Simulation")
    logger.debug("Output = %s" % (fname))

    logger.debug("-" * 100)
    logger.debug("")


if __name__ == "__main__":

    arg_list = []
    for vehs in [3, 5, 7, 9]:
        for tws in [1, 2, 4, 8]:
            arg_list.append((50, vehs, tws, 1, calling_policy))
    with Pool(4) as p:
        p.starmap(multiaddress, arg_list)
