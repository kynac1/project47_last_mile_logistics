import os
import numpy as np
from project47.simulation import *
from project47.data import *
from project47.routing import *
from project47.multiday_simulation import *
from functools import reduce
from multiprocessing import Pool


def test_sample_generator():
    cd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    sample_data = os.path.join(cd, "Toll_CHC_November_Sample_Data.csv")
    CHC_data = os.path.join(cd, "christchurch_street.csv")
    sample_df, CHC_df, CHC_sub, CHC_sub_dict = read_data(
        sample_data,
        CHC_data,
        lat_min=-43.6147000,
        lat_max=-43.4375000,
        lon_min=172.4768000,
        lon_max=172.7816000,
    )

    def sample_generator(rg: np.random.Generator):
        lat, lon = get_sample(
            10, rg, cd, sample_df, CHC_df, CHC_sub, CHC_sub_dict, save=False
        )
        time_windows = np.zeros((len(lat), 2))
        for i in range(len(lat)):
            time_windows[i, 0] = 0
            time_windows[i, 1] = 28800
        customers = [Customer(lat[i], lon[i], 0.9, 0.9, rg=rg) for i in range(len(lat))]

        return customers, time_windows

    return sample_generator


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
    depots, dm, tm, time_windows, day, arrival_days, futile_count, alternate_locations
):
    locs = dm.shape[0]
    r = ORToolsRouting(locs, 5)
    dim, ind = r.add_dimension(dm, 0, 50000, True, "distance")
    r.routing.SetArcCostEvaluatorOfAllVehicles(ind)
    dim, ind = r.add_time_windows(tm, time_windows, 28800, 28800, True, "time")
    for alternates in alternate_locations:
        r.add_option(alternates, 50000)

    s = r.solve(log=False)

    unscheduled = []
    scheduled = reduce(lambda x, y: x + y, s.routes)
    for i in range(locs):
        if i not in scheduled:
            unscheduled.append(i)
    return s, unscheduled


def base_sim(routes, dm, tm, delivery_time_windows, customers, rg: np.random.Generator):
    return sim(routes, base_policy(dm, tm, delivery_time_windows, customers, rg))


def test_multiday(
    sim_policy=base_policy,
    depots=np.array([[-43.5111688], [172.7319266]]),
    days=10,
    start_time=0,
    end_time=28800,
    seed=123456789,
    replications=1,
    arrival_rate=10,
    num_vehicles=5,
    max_distance=50000,
    drop_penalty=50000,
):
    cd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    sample_data = os.path.join(cd, "Toll_CHC_November_Sample_Data.csv")
    CHC_data = os.path.join(cd, "christchurch_street.csv")
    sample_df, CHC_df, CHC_sub, CHC_sub_dict = read_data(
        sample_data,
        CHC_data,
        lat_min=-43.6147000,
        lat_max=-43.4375000,
        lon_min=172.4768000,
        lon_max=172.7816000,
    )

    def sample_generator(rg: np.random.Generator):
        lat, lon = get_sample(
            rg.poisson(arrival_rate),
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
            time_windows[i, 0] = 0
            time_windows[i, 1] = 28800
        customers = [Customer(lat[i], lon[i], 0.9, 0.9, rg=rg) for i in range(len(lat))]

        return customers, time_windows

    def route_optimizer(
        depots,
        dm,
        tm,
        time_windows,
        day,
        arrival_days,
        futile_count,
        alternate_locations,
    ):
        locs = dm.shape[0]
        r = ORToolsRouting(locs, num_vehicles)
        dim, ind = r.add_dimension(dm, 0, max_distance, True, "distance")
        r.routing.SetArcCostEvaluatorOfAllVehicles(ind)
        dim, ind = r.add_time_windows(
            tm, time_windows, end_time, end_time, True, "time"
        )
        for alternates in alternate_locations:
            r.add_option(alternates, drop_penalty)

        s = r.solve(log=False)

        unscheduled = []
        scheduled = reduce(lambda x, y: x + y, s.routes)
        for i in range(locs):
            if i not in scheduled:
                unscheduled.append(i)
        return s, unscheduled

    def simulator(
        routes, dm, tm, delivery_time_windows, customers, rg: np.random.Generator
    ):
        return sim(routes, sim_policy(dm, tm, delivery_time_windows, customers, rg))

    return multiday(
        np.array([[-43.5111688], [172.7319266]]),
        sample_generator,
        dist_and_time,
        route_optimizer,
        simulator,
        days,
        start_time,
        end_time,
        seed=seed,
        replications=replications,
    )


def runner(kwargs):
    data = multiday(**kwargs)
    fname = 
    with open()

if __name__ == "__main__":
    pool = Pool()
