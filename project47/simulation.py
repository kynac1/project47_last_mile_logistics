
from project47.routing import *
import numpy as np
from copy import copy
import json
import os

def sim(s:RoutingSolution, distance_function, time_function, futile_function, windows, policies:list=[], seed:int=0):
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
            times[i] += time_function(route[j], route[j+1], times[i]) # arrivel time for job j+1
            isfail = False
            isfail = tw_policy1(i, j, route, distances, times, windows)
            # need to exclude the [0 0] case?
            # if futile_function(route[j], route[j+1], times[i]): # and route[j] != route[j+1]:
            #     isfail = True # to include failed delivery caused by other factors
            if isfail:
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

def tw_policy1(i, j, route, distances, times, windows):
    '''
    This time window policy makes the decision after arriving at the next place.
    The deliver man checks the time once arrived. 
    If the current time falls out of the time windows, then he will skip and go to the next place.
    '''
    # make the decision after arrival
    # out of time windows
    isfail = False
    print(times[i])
    if times[i] < windows[route[j+1]][0] or times[i] > windows[route[j+1]][1]:
        # skip j+1 job
        isfail = True
    return isfail

def tw_policy2(i, j, route, distances, times, windows, time_function):
    '''
    This time window policy makes the decision after arriving at the next place.
    The deliver man checks the time once arrived. 
    If the current time is earlier than the time windows, then he will wait till the availabe time.
    If the current time is later than the time windows, he will skip to next place.
    '''
    # make the decision after arrival
    # out of time windows
    if times[i] < windows[route[j+1]][0]:
        # add on the waiting time
        times[i] += windows[route[j+1]][0] - time_function(route[j], route[j+1], times[i])
    elif times[i] > windows[route[j+1]][0]:
        # skip j+1 job
        isfail = True
    return isfail
    




    



def collect_data(day, seed, s, distances, times, futile, delivered):
    '''
    day: day count over the week (i.e. 1-7)
    '''
    data = {}
    key = 'day'+str(day)+str(seed)
    data[key] = []
    time_deilvered = times[delivered]
    print(type(distances))
    data[key].append({
        "number_of_packages": len(futile)+len(delivered),
        "number_of_vehicles": len(s.routes), # is it number of used vehicles or just a total number?
        "distances": { i : distances[i] for i in range(0,len(distances))},
        "times": times.tolist(),
        "deliveries_attempted": '',#{[1,3,6,0,0]}, # successful deliveries or total deliveries?
        "futile_deliveries": { i : futile[i] for i in range(0,len(futile))},
        "delivered_packages": {
            "days_taken": '',#[1,4,2,7,32,2],
            "time_deilvered":  { i : time_deilvered[i] for i in range(0,len(time_deilvered))},#times[delivered],
            "time_window": ''#[[2,4],[2,3],[2,3],[1,4],[1,2]] 
            },
        "undelivered_packages": {
            "days_taken": '',#[12,54,21,43,21],
            "time_window": ''#[[3,7],[2,7],[5,9],[1,3],[4,5]] 
            }
    })

    cd = os.path.dirname(os.path.abspath(__file__)).strip('project47') + 'data'
    with open(os.path.join(cd,'simdata.json'), 'w') as outfile:
        json.dump(data, outfile, indent=2)
