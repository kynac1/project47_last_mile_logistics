from project47.routing import *
import numpy as np
from copy import copy
import json
import os
import gc

import logging
from typing import List

logger = logging.getLogger(__name__)


def sim(s: RoutingSolution, update_function):
    """Simple simulator

    TODO: Docs need a rewrite

    The current behaviour is to have each vehicle travel along each route individually. Times, distances,
    and the futile deliveries are calculated and recorded according to the provided functions.

    Vehicles are all assumed to leave at time 0, and travel without breaks.

    This can be seen as a discrete event simulation, where events occur at arrivals to each location.

    Parameters
    ----------
    s : RoutingSolution
        The solution for a single day for the routes each vehicle travels
    update_function
        Calculates the behaviour at each step

    Returns
    -------
    distances : np.array
        The distance each vehicle travels
    times : np.array
        The time each vehicle takes
    futile : np.array
        The number of futile deliveries for each vehicle
    delivered : list
        A list of all successful deliveries
    """
    logger.debug("Start simulation")
    times: List[List[int]] = []
    distances: List[List[int]] = []
    futile = np.zeros(len(s.routes))
    delivered = []

    for i, route in enumerate(s.routes):
        logger.debug("Vehicle %i" % i)
        times.append([])
        times[-1].append(0)
        distances.append([])
        distances[-1].append(0)

        j = 0

        logger.debug(
            "Route: %s, distance: %i, time: %i, futile: %s"
            % (
                route[j:],
                distances[-1][-1],
                times[-1][-1],
                futile[i],
            )
        )

        while j < (len(route) - 1):
            distance, time, isfutile, route_new = update_function(
                route, j, times[-1][-1]
            )
            distances[-1].append(distances[-1][-1] + distance)
            times[-1].append(times[-1][-1] + time)
            # compare two routes, update the route and the index
            if route != route_new:
                j = 0
                route = route_new
                # futile[i] += len()
                # delivered.append(route[j])
            else:
                if isfutile:
                    futile[i] += 1
                else:
                    if (
                        route[j + 1] != 0
                    ):  # Getting annoyed at all the depo nodes getting added here. We're only using 0 for depo, so this is fine as a quick hack
                        delivered.append(route[j + 1])
                j = j + 1

            logger.debug(
                "Route: %s, distance: %i, time: %i, futile: %s"
                % (
                    route[j:],
                    distances[-1][-1],
                    times[-1][-1],
                    futile[i],
                )
            )

    logger.debug("End simulation")
    return distances, times, futile, delivered


def default_update_function(distance_matrix, time_matrix, time_windows):
    """This should be seen as a basic example for testing. Should not use for actual simulation.

    This time window policy makes the decision after arriving at the next place.
    The deliver man checks the time once arrived.
    If the current time falls out of the time windows, then he will skip and go to the next place.
    """
    f = default_distance_function(distance_matrix)
    g = default_time_function(time_matrix)

    def h(route, i, time):
        next_distance = f(route[i], route[i + 1], time)
        next_time = g(route[i], route[i + 1], time)
        if route[i + 1] in time_windows:
            futile = (
                time + next_time < time_windows[route[i + 1]][0]
                or time + next_time > time_windows[route[i + 1]][1]
            )
        else:
            futile = True  # why futile is true if it does not have a tw?
        return next_distance, next_time, futile, route

    return h


def base_policy(
    distance_matrix,
    time_matrix,
    time_windows,
    customers,
    rg=np.random.Generator(np.random.PCG64(123)),
):
    """Does the most basic behaviour possible

    Travels to each location without rechecking anything, and no special behaviour for futile deliveries.

    Parameters
    ----------
    distance_matrix : np.array
        nxn matrix of distances
    time_matrix : np.array
        nxn matrix of times
    time_windows : np.array
        nx2 matrix of time windows. First column is time window starts, second is ends.
    customers : list
        List of customer objects, with a visit method
    rg : np.RandomGenerator
        Controls the stream of random numbers

    Returns
    -------
    function
        This is a closure, so it returns an update function that can be used in the `sim` function
    """

    f = default_distance_function(distance_matrix)
    g = default_time_function(time_matrix)

    def h(route, i, time):
        next_distance = f(route[i], route[i + 1], time)
        next_time = g(route[i], route[i + 1], time)
        futile = not customers[route[i + 1]].visit(
            time + next_time
        )  # check if the next delivery is going to be futile
        return next_distance, next_time, futile, route

    return h


def wait_policy(
    distance_matrix,
    time_matrix,
    time_windows,
    customers,
    rg=np.random.Generator(np.random.PCG64(123)),
):
    """Does the most basic behaviour possible

    Travels to each location without rechecking anything, and no special behaviour for futile deliveries.

    Parameters
    ----------
    distance_matrix : np.array
        nxn matrix of distances
    time_matrix : np.array
        nxn matrix of times
    time_windows : np.array
        nx2 matrix of time windows. First column is time window starts, second is ends.
    customers : list
        List of customer objects, with a visit method
    rg : np.RandomGenerator
        Controls the stream of random numbers

    Returns
    -------
    function
        This is a closure, so it returns an update function that can be used in the `sim` function
    """

    f = default_distance_function(distance_matrix)
    g = default_time_function(time_matrix)

    def h(route, i, time):
        next_distance = f(route[i], route[i + 1], time)
        next_time = g(route[i], route[i + 1], time)
        if time + next_time < time_windows[route[i + 1]][0]:
            next_time = time_windows[route[i + 1]][0] - time
        futile = not customers[route[i + 1]].visit(
            time + next_time
        )  # check if the next delivery is going to be futile
        return next_distance, next_time, futile, route

    return h


def truewait_policy(
    distance_matrix,
    time_matrix,
    time_windows,
    customers,
    rg=np.random.Generator(np.random.PCG64(123)),
):
    """ Some experimental stuff, may make more sense if it works?"""

    f = default_distance_function(distance_matrix)
    g = default_time_function(time_matrix)

    def h(route, i, time):
        next_distance = f(route[i], route[i + 1], time)
        next_time = g(route[i], route[i + 1], time)
        futile = not customers[route[i + 1]].visit(time + next_time)
        while time + next_time < time_windows[route[i + 1]][0]:
            next_time += 5
            futile = not customers[route[i + 1]].visit(time + next_time)
            if not futile:
                break
        return next_distance, next_time, futile, route

    return h


def estimate_ahead_policy(
    distance_matrix,
    time_matrix,
    time_windows,
    customers,
    rg=np.random.Generator(np.random.PCG64(123)),
):
    """Does the most basic behaviour possible

    Checks for lateness at each stage. If late, reroute. Should basically just remove the next location, but some reordering of other locations may occur.

    If a new route is returned in the update, no distance or time should elapse, and the current location should be at the start of the route.

    Parameters
    ----------
    distance_matrix : np.array
        nxn matrix of distances
    time_matrix : np.array
        nxn matrix of times
    time_windows : np.array
        nx2 matrix of time windows. First column is time window starts, second is ends.
    customers : list
        List of customer objects, with a visit method
    rg : np.RandomGenerator
        Controls the stream of random numbers

    Returns
    -------
    function
        This is a closure, so it returns an update function that can be used in the `sim` function
    """

    f = default_distance_function(distance_matrix)
    g = default_time_function(time_matrix)
    customers_list = customers.tolist()
    alternates = {
        i: [
            customers_list.index(a)
            for a in customers[i].alternates
            if a != customers[i]
        ]
        for i in range(len(customers))
    }

    def h(route, i, time):
        next_distance = f(route[i], route[i + 1], time)
        next_time = g(route[i], route[i + 1], time)
        if time + next_time < time_windows[route[i + 1]][0]:
            next_time = time_windows[route[i + 1]][0] - time
        if time + next_time > time_windows[route[i + 1]][1]:
            # go straight to depot if the next place is depot after skipping
            if False:  # route[i + 2] == 0:
                next_distance = f(route[i], route[i + 2], time)
                next_time = g(route[i], route[i + 2], time)
                route = [0]
                futile = True  # manually set to True to avoid appending depot to the delivery list in the simulation.
                # This should actually be unnecessary. We can strip those out after the simulation.
                return next_distance, next_time, futile, route
            # skip i+1 job and reroute
            else:
                logger.debug("Late for next delivery")
                route = rerouting_new(
                    i,
                    route,
                    distance_matrix,
                    time_matrix,
                    time_windows,
                    time,
                    alternates,
                )
                # print(route)
                next_distance = 0  # f(route[0], route[1], time)
                next_time = 0  # g(route[0], route[1], time)
                futile = 0  # not customers[route[1]].visit(time + next_time)
        else:
            futile = not customers[route[i + 1]].visit(time + next_time)

        return next_distance, next_time, futile, route

    return h


def calling_policy(distance_matrix, time_matrix, time_windows, customers, rg):
    """Does the most basic behaviour possible

    Calls the next customer at each stage. If customer is unresponsive, reroute.
    Should basically just remove the next location, but some reordering of other locations may occur.

    There's also some stuff here for waiting if a delivery is futile; not too sure about it though.

    If a new route is returned in the update, no distance or time should elapse, and the current location should be at the start of the route.

    Parameters
    ----------
    distance_matrix : np.array
        nxn matrix of distances
    time_matrix : np.array
        nxn matrix of times
    time_windows : np.array
        nx2 matrix of time windows. First column is time window starts, second is ends.
    customers : list
        List of customer objects, with a visit method, and a call_ahead method.
    rg : np.RandomGenerator
        Controls the stream of random numbers

    Returns
    -------
    function
        This is a closure, so it returns an update function that can be used in the `sim` function
    """

    f = default_distance_function(distance_matrix)
    g = default_time_function(time_matrix)
    customers_list = customers.tolist()
    alternates = {
        i: [
            customers_list.index(a)
            for a in customers[i].alternates
            if a != customers[i]
        ]
        for i in range(len(customers))
    }
    time_windows = time_windows

    def h(route, i, time):
        next_distance = f(route[i], route[i + 1], time)
        next_time = g(route[i], route[i + 1], time)
        if time + next_time < time_windows[route[i + 1]][0]:
            next_time = time_windows[route[i + 1]][0] - time
        if (time + next_time > time_windows[route[i + 1]][1]) or not customers[
            route[i + 1]
        ].call_ahead(time + next_time):
            # go straight to depot if the next place is depot after skipping
            if False:  # route[i + 2] == 0:
                next_distance = f(route[i], route[i + 2], time)
                next_time = g(route[i], route[i + 2], time)
                route = [0]
                futile = True  # manually set to True to avoid appending depot to the delivery list in the simulation.
                # This should actually be unnecessary. We can strip those out after the simulation.
                return next_distance, next_time, futile, route
            # skip i+1 job and reroute
            else:
                logger.debug("Late for delivery, or customer unresponsive")
                time_windows
                time_windows[route[i + 1], 0] = 0
                time_windows[route[i + 1], 1] = 1
                route = rerouting_new(
                    i,
                    route,
                    distance_matrix,
                    time_matrix,
                    time_windows,
                    time,
                    alternates,
                )
                next_distance = 0  # f(route[0], route[1], time)
                next_time = 5  # g(route[0], route[1], time)
                futile = 0  # not customers[route[1]].visit(time + next_time)
        else:

            futile = not customers[route[i + 1]].visit(time + next_time)

        return next_distance, next_time, futile, route

    return h


def new_tw_policy(distance_matrix, time_matrix, time_windows, customers, rg):
    """Idea is to get new time-windows from the customer, instead of simply calling ahead

    Doesn't quite work properly yet.
    """
    f = default_distance_function(distance_matrix)
    g = default_time_function(time_matrix)
    customers_list = customers.tolist()
    alternates = {
        i: [
            customers_list.index(a)
            for a in customers[i].alternates
            if a != customers[i]
        ]
        for i in range(len(customers))
    }
    time_windows = time_windows

    def h(route, i, time):
        next_distance = f(route[i], route[i + 1], time)
        next_time = g(route[i], route[i + 1], time)
        if time + next_time < time_windows[route[i + 1]][0]:
            next_time = time_windows[route[i + 1]][0] - time
        if time + next_time > time_windows[route[i + 1]][1]:
            logger.debug("Late for delivery")
            time_windows[route[i + 1], :] = customers[route[i + 1]].call_ahead_tw(
                time, options=[[0, 1], [time + next_time, 28800]]
            )
            route = rerouting_new(
                i,
                route,
                distance_matrix,
                time_matrix,
                time_windows,
                time,
                alternates,
            )
            next_distance = 0  # f(route[0], route[1], time)
            next_time = 5  # g(route[0], route[1], time)
            futile = 0  # not customers[route[1]].visit(time + next_time)
        else:

            futile = not customers[route[i + 1]].visit(time + next_time)
            if futile:
                logger.debug("Delivery Futile")
                time_windows[route[i + 1], :] = customers[route[i + 1]].call_ahead_tw(
                    time + next_time, options=[[0, 1], [time + next_time, 28800]]
                )
                route = rerouting_new(
                    i + 1,
                    route,
                    distance_matrix,
                    time_matrix,
                    time_windows,
                    time,
                    alternates,
                )
                next_time += 5

        return next_distance, next_time, futile, route

    return h


def default_distance_function(distance_matrix):
    """Basically makes an index into a function call. Might be useful if
    we want to swap in some randomness at some stage
    """

    def f(i, j, time):
        return distance_matrix[i, j]

    return f


def default_time_function(time_matrix):
    """See above. Could probably merge these actually."""

    def f(i, j, time):
        return time_matrix[i, j]

    return f


def rerouting(
    i, route, distance_matrix, time_matrix, time_windows, current_time=0, alternates={}
):
    """This is actually mostly unused, see rerouting_new. Leaving here for reference.

    This function finds the optimal routes from the current location to depot 'O'.

    Parameters
    ---------------
    current start: current starting place
    distance_matrix: original distance matrix
    time_matrix: original time matrix
    time_windows: original time windows

    Returns
    ---------------
    route: an optimal route from the current location to depot 'O'
    """
    # dm = default_distance_function(distance_matrix)
    # tm = default_time_function(time_matrix)

    # if route[i+2] == 0:
    #     next_distance = f(route[i],route[i+2],time)
    #     next_time = g(route[i],route[i+2],time)

    # drop next location if current place is depot
    if i == 0:
        # solve the problem
        locs = time_matrix.shape[0]
        r = ORToolsRouting(locs, 1, 0)
        dim, ind = r.add_time_windows(time_matrix, time_windows, 1, 10, False, "time")
        r.routing.SetArcCostEvaluatorOfAllVehicles(ind)
        r.add_disjunction(route[i + 1], 0)
        s = r.solve()
        if s is None:
            # Rerouting failed. Just return old route
            return route
        route_n = s.routes[0]
    else:
        # get the rest of the places that need to visit
        places_to_visit = route[i + 2 : :]
        places_to_visit.append(route[i])
        places_to_visit.sort()
        keys = list(np.arange(len(places_to_visit)))
        # record both the places that their indices in rerouting
        places_to_visit_dic = dict(zip(keys, places_to_visit))
        # current place - starting place for rerouting
        k = places_to_visit.index(route[i])

        # slice the distances for the places to visit
        dm = distance_matrix[places_to_visit]
        dm = dm[:, places_to_visit]
        # slice the times for the places to visit
        tm = time_matrix[places_to_visit]
        tm = tm[:, places_to_visit]
        # slice the time windows for the places to visit
        tw = time_windows[places_to_visit]

        # add tw for arbitrary depot
        tw = np.vstack((tw, np.array([0.0, 99999999999999.0])))
        # compute rerouting time matrix
        tm = rerouting_matrix(k, tm)
        # compute rerouting distance matrix
        dm = rerouting_matrix(k, dm)
        # print ("time_matrix", str(tm)) # printing result
        locs = tm.shape[0]
        depo = tm.shape[0] - 1

        # solve the problem
        r = ORToolsRouting(locs, 1, depo)
        dim, ind = r.add_time_windows(
            tm, tw, 28800, 28800, False, "time"
        )  # Previous bound and slack seem like a bug?
        r.routing.SetArcCostEvaluatorOfAllVehicles(ind)
        s = r.solve(tlim=1)
        if s is None:
            # Rerouting failed. Just return old route
            return route
        route_new = s.routes[0][1:-1]

        # route_n = places_to_visit_dic[route_new]
        route_n = [places_to_visit_dic[x] for x in route_new]

    return route_n


def rerouting_matrix(k, matrix):
    """
    This function computes the dm or tm needed for rerouting.

    Parameters
    ---------------
    k: current starting place
    matrix: original distance matrix or time matrix

    Returns
    ---------------
    matrix: dm or tm with an arbitrary depot
    """
    M = 99999999999999
    # add an arbitrary depot 'E'
    row_to_be_added = np.ones(matrix.shape[0]) * M
    # Adding row to numpy array
    matrix = np.vstack((matrix, row_to_be_added))
    col_to_be_added = np.ones(matrix.shape[0]) * M
    # Adding col to numpy array
    matrix = np.hstack((matrix, np.atleast_2d(col_to_be_added).T))
    # force an arc from 'E' to current place - to start from 'E'
    matrix[-1][k] = 0
    # force an arc from depot 'O' to 'E' - to arrive at 'O'
    matrix[0][-1] = 0

    return matrix


def rerouting_new(
    i,
    route,
    distance_matrix,
    time_matrix,
    time_windows,
    current_time,
    alternates={},
    tlim=5,
    fss=routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
    lsm=routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
    return_obj=False,
):
    """This function finds the optimal routes from the current location to the end of the route.


    Parameters
    ---------------
    i : int
        The current position of the vehicle along the route. route[i] is the current location.
    route : array-like
        The list of locations we are travelling along
    distance_matrix : array-like
        nxn, current distance matrix
    time_matrix : array-like
        nxn, current time matrix
    time_windows : array-like
        nx2, current time windows
    current_time : int
        The current time of day when we're at location i
    alternates : dict
        Maps each location index to all other location indices that can be used instead.
        So {1:[2,3]} says that location 1 can be replaced with locations 2 or 3.
        Obviously only one ends up being in the final route.

    Returns
    -------
    route : list
        The updated route. If solving fails, then returns route[i:] as the new route, without removing or changing locations.
    """
    logger.debug("Rerouting started")
    places_to_visit = np.array(
        route[i:]
    )  # I'm not removing locations yet. Quicker to implement that way.
    # Seems like changing the time window input and recomputing is a more general approach as well, which allows for more reorderings, and
    # easier to support alternate locations.

    for k, v in alternates.items():
        if (
            k in route[i + 1 : -1]
            and len(v) > 0  # index route to ignore starts and ends
        ):  # Need the length check; appending empty list does datatype conversion
            places_to_visit = np.append(places_to_visit, v)
    places_to_visit = places_to_visit.tolist()

    # slice the distances for the places to visit
    dm = distance_matrix[places_to_visit]
    dm = dm[:, places_to_visit]
    # slice the times for the places to visit
    tm = time_matrix[places_to_visit]
    tm = tm[:, places_to_visit]
    # slice the time windows for the places to visit
    # Also subtract current time
    tw = np.clip(time_windows[places_to_visit] - current_time, 0, np.inf)

    # The start location is route[i], which becomes 0 in the subset
    # The end location is route[-1]. Will be the size of the submatrix - 1 (cause 0 based indexing), as it's the last location that gets indexed.
    # This assumes the end location doesn't change, which makes sense.
    N = tw.shape[0]
    r = ORToolsRouting(N, 1, starts=[0], ends=[len(route[i:]) - 1])

    dim, ind = r.add_time_windows(
        tm, tw, 28800, 28800 - current_time, False, "time"
    )  # Assumes the default option of daylength = 28800
    r.routing.SetArcCostEvaluatorOfAllVehicles(ind)

    for j in range(1, len(route[i:]) - 1):  # Don't include start or end nodes
        original_node = places_to_visit[j]
        if original_node in alternates:  # Has alternate locations
            new_alternate_list = [j]
            for k in alternates[original_node]:
                new_alternate_list.append(places_to_visit.index(k))
            r.add_option(new_alternate_list, 10000)
        else:
            r.add_disjunction(j, 10000)

    # I've worked out these parameters are generally the fastest. Greedy descent is problematic in that it
    # can't escape local optima, but we should generally be close enough to optimal that it doesn't matter.
    r.search_parameters.first_solution_strategy = fss
    r.search_parameters.local_search_metaheuristic = lsm
    r.search_parameters.use_cp_sat = True
    s = r.solve(
        tlim=tlim, log=logger.getEffectiveLevel() <= 0
    )  # This solves the problem, logging if level is debug or less
    if s is None:
        logger.warning("Rerouting Failed")
        # Rerouting failed. Just return old route
        res = route[i:]
    else:
        logger.debug("Rerouting Successful")
        res = [
            places_to_visit[i] for i in s.routes[0]
        ]  # I think this does the same as previously? Not too sure. Makes sense though
    obj = r.objective
    del r
    del s
    gc.collect()
    if return_obj:
        return obj
    else:
        return res


import time

if __name__ == "__main__":
    n = 100
    tw = np.zeros((n, 2))
    tw[:, 1] = np.ones(n) * 100
    start = time.time()

    print(
        rerouting_new(
            20,
            [i for i in range(n)] + [0],
            np.ones((n, n)) + np.random.rand(n, n) * 20,
            np.ones((n, n)) + np.random.rand(n, n) * 20,
            tw,
            20,
            {i + 20: [i] for i in range(1, 19)},
        )
    )

    print(time.time() - start)
