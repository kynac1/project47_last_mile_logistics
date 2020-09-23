import numpy as np
from project47.routing import *
from project47.customer import Customer
from project47.data import get_sample, read_data
from numpy.random import Generator, PCG64
import os
import matplotlib.pyplot as plt

# from celluloid import Camera

import logging

logger = logging.getLogger(__name__)


def collect_data(
    day: int,
    solution: RoutingSolution,
    distances: list,
    times: list,
    futile: np.array,
    delivered: np.array,
    arrival_days: list,
    time_windows: dict,
):
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
        "distances": [int(sum(veh_dists)) for veh_dists in distances],
        "times": [int(sum(veh_times)) for veh_times in times],
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
):
    """Multiday Sim

    Paramters
    ---------
    depots : np.array
        2*n_depots array of longitudes and latitudes of the depots.
        This is set up to support multidepot problems. However, to do this properly we'll need to track which depots
        have which packages. Need to think about this more.
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
    replications : int
        Defaults to 1. The number of simulations to perform on the optimized route. Only the last is used as the input to the next day.
        (Might be an idea to take the mode values if enough simulations are performed?)
    plot : bool
        Whether to display a plot of the current routes
    """

    logger.debug("Start multiday sim")

    rg = Generator(PCG64(seed))

    # Pregenerate arrivals
    latitudes_per_day = []
    longitudes_per_day = []
    time_windows_per_day = []
    customers_per_day = []

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

    for day in range(n_days):
        logger.debug("Start day %i" % day)

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

        ## TODO: Remove packages from collection points
        if collection_points:
            # initialise a list of dictionaries for each collection point
            packages_at_collection = [{} for i in range(k)]
            for i in range(k):
                if len(packages_at_collection[i]) != 0:
                    # randome number of customers collecting today
                    rd2 = rg.integers(low=0, high=len(packages_at_collection[i]), size=1)
                    arrivals = np.random.poisson(1, size=rd2)
                    # sort the pakacges in order of descending days in collection point
                    packages_at_collection[i] = sorted(packages_at_collection[i].items(), key=lambda x: x[1], reverse=True)
                    for j in range(rd2):
                        if rg.random(low=0, high=1, size=1) > 0.8:
                            collected_package = packages_at_collection[i].pop(packages_at_collection[i].keys()[len(packages_at_collection[i])-j])
                        else:
                            collected_package = packages_at_collection[i].pop(packages_at_collection[i].keys()[0])
                        
                        # add to delivered package?
                    # max_value = max(packages_at_collection[i].values())
                    # max_keys = [k for k, v in packages_at_collection[i].items() if v == max_value] # getting all keys containing the `maximum
            # collected_packages = 
            #     collected_packages, cost = collection_points(
            #         unscheduled, undelivered_packages, futile_count, dm, tm, customers, rg
            #     )
        ## TODO: Add customers to collections points, and add visited collection points to customers
        # need a list of undelivered packages in the simulation
        allocat_packages_to_collection = [[] for i in range(k)]
        for i, c in enumerate(undelivered_packages):
            # a threshold of the number of days for the package in the system
            if undelivered_packages["days_taken"][i] > 5:
                # get the dist from the cusomter's house to the collection points
                source = np.arange(len(fac_lat))
                lat_all = sol_fac_lat
                lon_all, = sol_fac_lon
                lat_all.insert(0,c.lat)
                lon_all,.insert(0,c.lon)
                dist, tm = osrm_get_dist(
                    cd,
                    coord_filename,
                    lat_all,
                    lon_all,
                    source = ,
                    save=False,
                    host="0.0.0.0:5000",)
                # choose the closest collection point
                min_value = min(dist[0])
                # allow the package to be assigned to the closest collection point if the dist is within the threshold
                if min_value < dist_threshold:
                    min_ind = lt.index(min_value)
                    # assign the package to its closest collection point
                    allocat_packages_to_collection[min_ind].append(c)
        # TODO: add one day to all packages_at_collection
        for i in range(k): 
            for j in len(allocat_packages_to_collection[i]):
                # if the collection point exceeds its capacity
                if len(packages_at_collection[i]) >= cap:
                    break
                packages_at_collection[i].append(allocat_packages_to_collection[i][j]) # Add customers to collections points
                # assign the appeded package with the number of days in cp of 0
        ## TODO: Remove packages sent to collection points from customers
        ## TODO: Output Data Correction
        # set the packages sent to collection points as delivered package
        # change futile to False?


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

        logger.debug(routes)
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
                )
            )

        # Remove delivered packages, using just the last result
        undelivered = np.ones(len(customers), dtype=bool)
        for alternates in alternate_locations:  # Remove all alternate locations as well
            for package in delivered:
                if package in alternates:
                    undelivered[alternates] = False
        undelivered[[i for i in range(n_depots)]] = True
        delivery_time_windows = delivery_time_windows[undelivered]
        arrival_days = arrival_days[undelivered]
        futile_count = futile_count[undelivered]
        customers = customers[undelivered]

        logger.debug(
            "Number of remaining Packages: %i" % (len(customers) - 1)
        )  # -1 for depo

    return data


def collection_point_example(unscheduled, futile_count, dm, tm, customers, rg):
    
