import os
import numpy as np
from project47.simulation import *
from project47.data import *
from project47.routing import *
from project47.multiday_simulation import *
from functools import reduce
from multiprocessing import Process

import logging

logging.basicConfig(filemode="w", level="DEBUG")
logger = logging.getLogger(__name__)


def no_time_windows_comparison(arrival_rate, num_vehicles, num_time_windows):
    logger.debug("Starting Experiment:")
    logger.debug(
        "Arrival rate = %s, No. Vehicles = %s, No. Time Windows = %s"
        % (arrival_rate, num_vehicles, num_time_windows)
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
            arrival_rate,  # rg.poisson(arrival_rate),
            rg,
            cd,
            sample_df,
            CHC_df,
            CHC_sub,
            CHC_sub_dict,
            save=False,
        )
        time_windows = np.zeros((len(lat), 2))
        for i in range(len(lat)):
            interval = (day_end - day_start) / num_time_windows
            for j in range(num_time_windows):
                if rg.random() > (num_time_windows - (j + 1)) / num_time_windows:
                    time_windows[i, 0] = interval * j
                    time_windows[i, 1] = interval * (j + 1)
                    break

        customers = [Customer(lat[i], lon[i], 0.5, 0.5, rg=rg) for i in range(len(lat))]

        return customers, time_windows

    def dist_and_time(customers):
        return osrm_get_dist(
            "",
            "",
            [customer.lat for customer in customers],
            [customer.lon for customer in customers],
            host="localhost:5000",
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
        tlim=10,
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
        s = r.solve(tlim=tlim, log=False)

        unscheduled = []
        scheduled = reduce(lambda x, y: x + y, s.routes)
        for i in range(locs):
            if i not in scheduled:
                unscheduled.append(i)
        return s, unscheduled

    def simulator(
        routes, dm, tm, delivery_time_windows, customers, rg: np.random.Generator
    ):
        return sim(
            routes, calling_policy(dm, tm, delivery_time_windows.copy(), customers, rg)
        )

    data = multiday(
        np.array([[-43.5111688], [172.7319266]]),
        sample_generator,
        dist_and_time,
        route_optimizer,
        simulator,
        100,
        day_start,
        day_end,
        plot=False,
        seed=2123897,
    )

    with open(
        f"experiments/results/experiments_{num_time_windows}_{arrival_rate}_{num_vehicles}.json",
        "w",
    ) as outfile:
        json.dump(data, outfile, separators=(",", ":"))


if __name__ == "__main__":
    no_time_windows_comparison(10, 5, 2)
    """
    try:
        with open(
            f"experiments/results/experiments_4_time_windows_20_5.json", "w"
        ) as f:
            pass
        print("OK")
    except:
        print("Cannot open file location")
        exit()
    for rate in [10, 20]:
        for vehs in [3, 5]:
            print(f"Running {rate}, {vehs}")
            p1 = Process(target=no_time_windows_comparison, args=(rate, vehs, 1))
            p2 = Process(target=no_time_windows_comparison, args=(rate, vehs, 4))
            p1.start()
            p2.start()
            p1.join()
            p2.join()
    """
