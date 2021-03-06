import numpy as np
from project47.routing import *
from project47.customer import Customer
from project47.data import get_sample, read_data
from project47.flp_data import *
from numpy.random import Generator, PCG64
import os
import matplotlib.pyplot as plt
from itertools import islice
from scipy.stats import geom
import time

# from celluloid import Camera

import logging

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


def collect_data(
    day: int,
    solution: RoutingSolution,
    distances: list,
    times: list,
    futile: np.array,
    delivered: np.array,
    arrival_days: list,
    time_windows: dict,
    collection_point_packages: list,
    collection_point_removed_packages: list,
    collection_dist: int,
):
    """Takes the data for a single day and structures it in a dictionary so that we can save it as json

    Parameters
    ----------
    day : int
        The current day number
    solution : RoutingSolution
        What the VRP solver returned for the routes
    distances : list
        The distance traveled for each vehicle
    times : list
        The time each vehicle took
    futile : list
        The number of futile deliveries each vehicle had
    arrival_days : list
        The day each customer came into the system
    time_windows : dict
        Maps the package number to the known time-window
    collection_point_packages : list
        The number of packages at each collection point
    collection_point_removed_packages : list
        The number of packages removed from each collection point
    collection_dist : list
        The distance travelled by customers to get to the collection point
    """
    data = {}
    """
    time_delivered = []
    for d in delivered:
        if d != 0:
            found = False
            for i, route in enumerate(solution.routes):
                for j, node in enumerate(route):
                    if node == d:
                        time_delivered.append(times[i][j])
                        found = True
                        break
                if found:
                    break
            if not found:
                time_delivered.append(
                    -1
                )  # Error. Hopefully negative time is obviously wrong."""

    data = {
        "day": day,
        "number_of_packages": int(
            len(arrival_days) - 1
        ),  # This could potentially be higher than the number of deliveries in the routes
        "number_of_vehicles": int(
            len(solution.routes)
        ),  # is it number of used vehicles or just a total number? ## Either way, I think we don't actually need this; we can get it from other info
        "distances": [int(veh_dists[-1]) for veh_dists in distances],
        "times": [int(veh_times[-1]) for veh_times in times],
        "deliveries_attempted": [
            int(len(route) - 2) for route in solution.routes
        ],  # {[1,3,6,0,0]}, # successful deliveries or total deliveries? ## Attempted, so total. -2 for depo at start and end.
        "futile_deliveries": [int(f) for f in futile],
        "delivered_packages": {
            "days_taken": [
                int(day - arrival_days[i]) for i in delivered if i != 0
            ],  # [1,4,2,7,32,2],
            # "time_delivered": [int(t) for t in time_delivered],
            "time_window": [
                [int(time_windows[i][0]), int(time_windows[i][1])]
                for i in delivered
                if i != 0
            ],  # [[2,4],[2,3],[2,3],[1,4],[1,2]]
        },
        "undelivered_packages": {
            "days_taken": [
                int(day - arrival_days[i])
                for i in range(len(arrival_days))
                if i != 0 and i not in delivered
            ],  # [12,54,21,43,21],
            "time_window": [
                [int(time_windows[i][0]), int(time_windows[i][1])]
                for i in range(len(arrival_days))
                if i != 0 and i not in delivered
            ],  # [[3,7],[2,7],[5,9],[1,3],[4,5]]
        },
        "collection_point_packages": collection_point_packages,
        "collection_point_removed_packages": collection_point_removed_packages,
        "collection_dist": collection_dist,
    }

    return data


def multiday(
    depots,
    sample_generator,
    dist_and_time,
    route_optimizer,
    simulator,
    n_days,
    day_start,
    day_end,
    seed=None,
    replications=1,
    plot=False,
    collection_points=None,
    k=0,
    dist_threshold=20000,
    futile_count_threshold=1,
    cap=20,
    tlim=1e10,
):
    """Multiday Sim

    Paramters
    ---------
    depots : np.array
        2*n_depots array of longitudes and latitudes of the depots.
        This is set up to support multidepot problems. However, to do this properly we'll need to track which depots
        have which packages. So this isn't actually fully supported.
    sample_generator : function
        Takes no inputs, returns two lists, longitudes and latitudes of the packages.
    dist_and_time : function
        Takes longitudes and latitudes, and returns a distance matrix, a time matrix, and a array of time windows.
    route_optimizer : function
        Inputs are the depot numbers, the distance and time matrices, the time windows as a np.array,
        the current day, the day each package arrived, and the number of times each package was futile.
        Outputs are a set of vehicle routes, and a list of packages that were not scheduled.
    simulator : function
        Simulates the deliveries for a set of routes, given the routes, distances, times and time windows.
    n_days : int
        The number of days to simulate.
    day_start : int
        The time for the start of a day
    day_end : int
        The time for the end of a day
    seed : int (Optional)
        The seed to initialise the random number generator with
    replications : int (Optional)
        Defaults to 1. The number of simulations to perform on the optimized route. Only the last is used as the input to the next day.
        (Might be an idea to take the mode values if enough simulations are performed?)
    plot : bool (Optional)
        Whether to display a plot of the current routes.
    collection_points: bool (Optional)
        Whether to enable the use of collection points
    k : int (Optional)
        If we have collection points, the number of collection points
    dist_threshold : int
        The distance a customer needs to be within to have their package assigned to a collection point
    futile_count_threshold : int
        The number of times a package needs to be futile before it gets assigned to a collection point
    cap : int
        The capacity of the collection points
    tlim : int
        The time to run the simulation for. Helps us stop the simulation if we get an extreme buildup of packages
    """
    start = time.time()
    logger.debug("Start multiday sim")

    rg = Generator(PCG64(seed))

    # Pregenerate arrivals
    latitudes_per_day = []
    longitudes_per_day = []
    time_windows_per_day = []
    customers_per_day = []
    allocat_packages_to_collection = [[] for i in range(k)]  # preset package allocation
    customer_to_cp = [
        [] for i in range(k)
    ]  # initialise the customer list for each collection point

    logger.debug("Generating incoming packages")

    for day in range(n_days):
        customers, new_time_windows = sample_generator(rg)
        latitudes_per_day.append([c.lat for c in customers])
        longitudes_per_day.append([c.lon for c in customers])
        time_windows_per_day.append(new_time_windows)
        customers_per_day.append(customers)

    data = []
    n_depots = depots.shape[1]
    delivery_time_windows = np.array(
        [[day_start, day_end] for i in range(n_depots)]
    )  # These are our beliefs about the time windows, not their true value
    arrival_days = np.zeros(n_depots)
    futile_count = np.zeros(n_depots)
    customers = np.array(
        [
            Customer(depots[0, 0], depots[1, 0], 1, 1, rg=rg)
            for i in range(len(depots[0]))
        ]
    )
    packages_at_collection = []
    collection_point_removed_packages = 0
    if collection_points and k != 0:  # choose the number of collection points
        sol_fac_lat, sol_fac_lon, coord, fac_coord = opt_collection_coord(
            k, cap, depots, sample_generator, dist_and_time, seed=None
        )

        # initialise a list of dictionaries for each collection point
        packages_at_collection = [{} for i in range(k)]
        # collection_point_removed_packages = [0 for i in range(k)]
    for day in range(n_days):
        logger.debug("Start day %i" % day)
        collection_point_removed_packages = [0 for i in range(k)]
        # Generate data
        new_time_windows, new_customers = (
            time_windows_per_day[day],
            customers_per_day[day],
        )

        delivery_time_windows = np.vstack((delivery_time_windows, new_time_windows))
        arrival_days = np.append(arrival_days, [day for _ in range(len(new_customers))])
        futile_count = np.append(futile_count, np.zeros(len(new_customers)))
        customers = np.append(customers, new_customers)

        logger.debug("Number of incoming packages: %i" % len(new_customers))
        logger.debug(
            "Current number of packages: %i" % (len(customers) - 1)
        )  # -1 for depo

        logger.debug("Calculating distance and time matrix")

        cp_customers = np.array([])
        collection_dist = 0
        # Remove packages from collection points
        if collection_points and k != 0:
            for i in range(k):
                logger.debug(
                    "Number of packages in collection %i, day %i: %i",
                    i,
                    day,
                    len(packages_at_collection[i]),
                )
                if (
                    len(packages_at_collection[i]) != 0
                ):  # there's packages in the collection point
                    p = 0.6  # success probability
                    # np.random.geometric(p=0.35, size=10000)
                    # np.random.geometric(p=0.6, size=len(packages_at_collection[i]))
                    collected_package = []
                    for c in packages_at_collection[i]:
                        v = packages_at_collection[i][c]
                        # package collection distribution
                        cdf = geom.cdf(v, p)
                        # if the probablity is greater than the random number, the package is picked up
                        if cdf >= rg.random():
                            collected_package.append(c)
                            collection_point_removed_packages[i] += 1
                        else:
                            # add a day to the number of days at collection point
                            packages_at_collection[i][c] += 1
                    for collected in collected_package:
                        packages_at_collection[i].pop(collected)

            # Add customers to collections points, and add visited collection points to customers
            #  a list of undelivered packages in the simulation
            undelivered = np.ones(len(futile_count), dtype=bool)
            for i, c in enumerate(futile_count):
                # a threshold of day count of the package in the system
                if c >= futile_count_threshold and i >= n_depots:
                    cd = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), "..", "data"
                    )
                    # get the dist from the cusomter's house to the collection points
                    lat_all = sol_fac_lat[:]
                    lon_all = sol_fac_lon[:]
                    lat_all.insert(0, customers[i].lat)
                    lon_all.insert(0, customers[i].lon)
                    if len(lat_all) > 4:
                        print("error")
                    coord_filename = None
                    dist, tm = osrm_get_dist(
                        cd,
                        coord_filename,
                        lat_all,
                        lon_all,
                        source=[0],
                        save=False,
                        host="localhost:5000",
                    )
                    # choose the closest collection point
                    min_value = min(dist[0])
                    # dist_threshold = 20000  # 20km
                    # allow the package to be assigned to the closest collection point if the dist is within the threshold
                    if min_value < dist_threshold:
                        min_ind = dist[0].index(min_value)
                        # assign if its closet collection point still has spare space
                        if len(packages_at_collection[min_ind]) < cap:
                            # assign the package to its closest collection point
                            allocat_packages_to_collection[min_ind].append(
                                i
                            )  # i is the index but not the unique index for the customer?
                            undelivered[
                                i
                            ] = False  # customer to be removed from the delivery list
                            # record the customer list for each collection point
                            customer_to_cp[min_ind].append(customers[i])
                            collection_dist += min_value
                            # tw_to_cp[min_ind].append(delivery_time_windows[i])
                            # ad_to_cp[min_ind].append(arrival_days[i])
                            # fc_to_cp[min_ind].append(futile_count[i])

                            # packages_at_collection[min_ind][customers[i]] = 0
            # Remove packages sent to collection points from customers
            delivery_time_windows = delivery_time_windows[undelivered]
            arrival_days = arrival_days[undelivered]
            futile_count = futile_count[undelivered]
            customers = customers[undelivered]

            cp_customers = np.array([])
            # add collection point as a customer if there is package allocated to it
            for cp in range(k):
                if len(customer_to_cp[cp]) != 0:
                    cp_customers = np.append(
                        cp_customers,
                        Customer(sol_fac_lat[cp], sol_fac_lon[cp], 1, 1, rg=rg),
                    )
            # customers = np.append(customers, new_customers)

            # cp_customers = np.array(
            #     [
            #         Customer(sol_fac_lat[cp], sol_fac_lon[cp], 1, 1, rg=rg)
            #         for cp in range(k)
            #         if len(customer_to_cp[cp]) != 0
            #     ]
            # )
            if len(cp_customers) > 0:
                cp_time_windows = np.array(
                    [[day_start, day_end] for i in range(len(cp_customers))]
                )

                delivery_time_windows = np.vstack(
                    (delivery_time_windows, cp_time_windows)
                )
                customers = np.append(customers, cp_customers)
                futile_count = np.append(futile_count, np.zeros(len(cp_customers)))
                arrival_days = np.append(
                    arrival_days, [day for _ in range(len(cp_customers))]
                )

        # Get times and distances
        dm, tm = dist_and_time(customers)
        if dm is None:
            logger.critical("Distance computation failed. Stopping simulation.")
            # We've exceeded the map bounds. Stop here for now, but we should really handle this more gracefully.
            break
        dm = np.array(dm)
        tm = np.array(tm)

        logger.debug("Compute alternate locations")

        # Setup list of alternate locations
        alternate_locations = []
        temp = customers.tolist()
        while len(temp) > 0:
            c = temp.pop()
            location_index = []
            for a in c.alternates:
                location_index.append(customers.tolist().index(a))
                if a in temp:
                    temp.remove(a)
            alternate_locations.append(location_index)

        logger.debug("Optimise routes")

        # Calulate routes for the day
        routes, unscheduled = route_optimizer(
            [i for i in range(n_depots)],
            dm,
            tm,
            delivery_time_windows,
            day,
            arrival_days,
            futile_count,
            alternate_locations,
        )
        if plot:
            plt.clf()
            routes.plot(
                positions=[(customer.lon, customer.lat) for customer in customers],
                weight_matrix=dm,
            )
            plt.show(block=False)
            plt.pause(0.001)

        futile_count[[i for i in range(len(customers)) if i not in unscheduled]] += 1

        # logger.debug(routes)
        logger.debug("Unscheduled: %s" % unscheduled)

        logger.debug("Start simulations")

        for i in range(replications):
            logger.debug("Replication %i" % i)
            # Simulate behaviour
            distances, times, futile, delivered = simulator(
                routes, dm, tm, delivery_time_windows, customers, rg
            )
            logger.debug("Delivered: %s" % delivered)

            # Data collection to save
            data.append(
                collect_data(
                    day,
                    routes,
                    distances,
                    times,
                    futile,
                    delivered,
                    arrival_days,
                    delivery_time_windows,
                    [len(l) for l in packages_at_collection],
                    collection_point_removed_packages,
                    collection_dist,
                )
            )

        # Remove delivered packages, using just the last result
        undelivered = np.ones(len(customers), dtype=bool)
        for alternates in alternate_locations:  # Remove all alternate locations as well
            for package in delivered:
                if package in alternates:
                    undelivered[alternates] = False
        undelivered[[i for i in range(n_depots)]] = True

        # # get the undelivered list for collection point
        # cp_undelivered = undelivered[-len(cp_customers) :]
        # count_cp_undelivered = 0

        # check if collection point is visited
        for i in range(len(cp_customers)):
            if not undelivered[-i - 1]:  # collection point visited
                for cust in customer_to_cp[
                    -i - 1
                ]:  # add package into the collection point
                    packages_at_collection[-i - 1][cust] = 0
                # allocat_packages_to_collection[min_ind] = []
                customer_to_cp[
                    -i - 1
                ] = []  # reset the customer list for the visited collection point
            else:
                # count_cp_undelivered += 1
                undelivered[
                    -i - 1
                ] = False  # remove collection point in the customer list
                # Generate data
                # new_time_windows, new_arrival_days, new_futile_count, new_customers = (
                #     tw_to_cp[-i - 1],
                #     ad_to_cp[-i - 1],
                #     fc_to_cp[-i - 1],
                #     customer_to_cp[-i - 1],
                # )

        delivery_time_windows = delivery_time_windows[undelivered]
        arrival_days = arrival_days[undelivered]
        futile_count = futile_count[undelivered]
        customers = customers[undelivered]
        # if count_cp_undelivered:
        #     # stick the undelivered collection pacakges on
        #     delivery_time_windows = np.vstack((delivery_time_windows, new_time_windows))
        #     arrival_days = np.append(arrival_days, new_arrival_days)
        #     futile_count = np.append(futile_count, new_futile_count)
        #     customers = np.append(customers, new_customers)

        logger.debug(
            "Number of remaining Packages: %i" % (len(customers) - 1)
        )  # -1 for depo
        if time.time() - start > tlim:
            break

    return data


def collection_point_example(unscheduled, futile_count, dm, tm, customers, rg):
    pass
