
from project47.routing import *
import numpy as np
from copy import copy

def sim(s:RoutingSolution, distance_function, time_function, futile_function, policies:list=[], seed:int=0):
    """ Simple simulator

    The current behaviour is to have each vehicle travel along each route individually. Times, distances,
    and the futile deliveries are calculated and recorded according to the provided functions.

    TODO: Implement some handling of time windows. We can already capture the customers behaviour using the futile
    function, but how should we handle arriving too early, or knowing we'll arrive late?

    Vehicles are all assumed to leave at time 0, and travel without breaks.

    The behaviour can be altered somewhat by providing a policy function, which alters the solution according to
    the current state. This is somewhat experimental, and not all policies can be captured with this. But,
    a fair few may be; need to experiment with them more.
    On a related topic, I wonder if there would be any benefit to using a discrete event simulation library here?

    Parameters
    ----------
    s : RoutingSolution
        The solution for a single day for the routes each vehicle travels
    distance_function : function
        Given the location numbers and the current time, calculate the distance to travel
        between them.
    time_function : function
        Given the location numbers and the current time, calculate the time to travel
        between them.
    futile_function : function
        Given the location numbers and the current time, calculate whether the next delivery will be futile.
        This is a bit weird, it doesn't really fit in the same framework as distance and time.
    policies : list
        A list of functions to execute on the current solution each time we travel between locations.
    seed : int
        The random seed. Defaults to 0 for reproducibility. We should try to stick with numpy random functions
        throughout our code to assist with this.
    
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
    s = copy(s)
    np.random.seed(seed)
    times = np.zeros(len(s.routes))
    distances = np.zeros(len(s.routes))
    futile = np.zeros(len(s.routes))
    delivered = []

    for i,route in enumerate(s.routes):
        for j in range(len(route)-1):
            distances[i] += distance_function(route[j], route[j+1], times[i])
            times[i] += time_function(route[j], route[j+1], times[i])
            if futile_function(route[j], route[j+1], times[i]):
                futile[i] += 1
            else:
                delivered.append(route[j])

            for policy in policies:
                policy(s, route, j, times[i])
    
    return distances, times, futile, delivered

def default_distance_function(distance_matrix):
    def f(i,j,time):
        return distance_matrix[i,j]
    return f

def default_time_function(time_matrix):
    def f(i,j,time):
        return time_matrix[i,j]
    return f

def default_futile_function(prob):
    def f(i,j,time):
        return np.random.rand() < prob
    return f

