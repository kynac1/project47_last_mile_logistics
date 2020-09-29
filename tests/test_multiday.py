from project47.data import *
from project47.routing import *
from project47.simulation import *
from project47.multiday_simulation import *
from project47.customer import Customer
from functools import reduce

import logging

logging.basicConfig(level="DEBUG")  # Set the global logging level


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
            10, rg, cd, sample_df, CHC_df, CHC_sub, CHC_sub_dict, save=False
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
    tlim=10,
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
    r.search_parameters.use_cp_sat = False

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
    return sim(routes, new_tw_policy(dm, tm, delivery_time_windows, customers, rg))


def test_multiday():
    """This is the main example of all the functionality.

    The idea is that when we create a new experiment to run, we'd copy the structure of this function and replace
    parts so that it implements the new policies
    """
    sample_generator = test_sample_generator()

    data = multiday(
        np.array([[-43.5111688], [172.7319266]]),
        sample_generator,
        dist_and_time,
        route_optimizer,
        simulator,
        10,
        0,
        28800,
        seed=123456789,
        replications=2,
        plot=False,
        collection_points=True,
    )

    assert len(data) == 20


def test_reproducible():
    """Run two identical simulations with same random seed. Check they return the same results."""
    sample_generator = test_sample_generator()

    data1 = multiday(
        np.array([[-43.5111688], [172.7319266]]),
        sample_generator,
        dist_and_time,
        route_optimizer,
        simulator,
        10,
        0,
        28800,
        seed=123456789,
    )

    data2 = multiday(
        np.array([[-43.5111688], [172.7319266]]),
        sample_generator,
        dist_and_time,
        route_optimizer,
        simulator,
        10,
        0,
        28800,
        seed=123456789,
    )

    assert data1 == data2


def test_alternate_locations():

    cd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    sample_data = os.path.join(cd, "Toll_CHC_November_Sample_Data.csv")
    CHC_data = os.path.join(cd, "christchurch_street.csv")
    sample_df, CHC_df, _, CHC_sub, CHC_sub_dict = read_data(sample_data, CHC_data)

    def sample_generator(rg: np.random.Generator):
        lat, lon = get_sample(
            20, rg, cd, sample_df, CHC_df, CHC_sub, CHC_sub_dict, save=False
        )
        time_windows = np.zeros((len(lat), 2))
        for i in range(len(lat)):
            time_windows[i, 0] = 0
            time_windows[i, 1] = 28800
        customers = [Customer(lat[i], lon[i], 0.9, 0.9, rg=rg) for i in range(len(lat))]
        for i in range(10):
            customers[i].add_alternate(customers[i + 10])

        return customers, time_windows

    data = multiday(
        np.array([[-43.5111688], [172.7319266]]),
        sample_generator,
        dist_and_time,
        route_optimizer,
        simulator,
        1,
        0,
        28800,
        replications=5,
    )


"""
def test_sim_performance():
    sample_generator = test_sample_generator()

    data = multiday(
        np.array([[-43.5111688], [172.7319266]]),
        sample_generator,
        dist_and_time,
        route_optimizer,
        simulator,
        2,
        0,
        28800,
        replications=1000,
    )

    assert len(data) == 2000"""


def test_plot():
    sample_generator = test_sample_generator()

    multiday(
        np.array([[-43.5111688], [172.7319266]]),
        sample_generator,
        dist_and_time,
        route_optimizer,
        simulator,
        1,
        0,
        28800,
        seed=293462,
        replications=1,
        plot=True,
    )
    plt.show()


if __name__ == "__main__":
    test_multiday()
    # test_alternate_locations()
