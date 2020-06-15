
from project47.routing import *
import numpy as np
from copy import copy
import json
import os

def sim(s:RoutingSolution, update_function, start_times={},seed:int=0):
    """ Simple simulator

    TODO: Docs need a rewrite

    The current behaviour is to have each vehicle travel along each route individually. Times, distances,
    and the futile deliveries are calculated and recorded according to the provided functions.

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
    times = []
    distances = []
    futile = np.zeros(len(s.routes))
    delivered = []

    for route, time in start_times:
        times[route] = time

    for i,route in enumerate(s.routes):
        times.append([])
        times[-1].append(0)
        distances.append([])
        distances[-1].append(0)
        for j in range(len(route)-1):
            distance, time, isfutile = update_function(route, j, times[-1][-1])
            distances[-1].append(distances[-1][-1] + distance)
            times[-1].append(times[-1][-1] + time)
            if isfutile:
                futile[i] += 1
            else:
                delivered.append(route[j])
                
    return distances, times, futile, delivered

def default_update_function(distance_matrix, time_matrix, time_windows):
    '''
    This time window policy makes the decision after arriving at the next place.
    The deliver man checks the time once arrived. 
    If the current time falls out of the time windows, then he will skip and go to the next place.
    '''
    f = default_distance_function(distance_matrix)
    g = default_time_function(time_matrix)
    def h(route, i, time):
        next_distance = f(route[i],route[i+1],time)
        next_time = g(route[i],route[i+1],time)
        if route[i+1] in time_windows:
            futile = time+next_time < time_windows[route[i+1]][0] or time+next_time > time_windows[route[i+1]][1]
        else:
            futile = True
        return next_distance, next_time, futile

    return h

def update_function2(distance_matrix, time_matrix, time_windows):
    '''
    This time window policy makes the decision after arriving at the next place.
    The deliver man checks the time once arrived. 
    If the current time is earlier than the time windows, then he will wait till the availabe time.
    If the current time is later than the time windows, he will skip to next place.
    '''

    f = default_distance_function(distance_matrix)
    g = default_time_function(time_matrix)
    def h(route, i, time):
        next_distance = f(route[i],route[i+1],time)
        next_time = g(route[i],route[i+1],time)
        if route[i+1] in time_windows:
            if time+next_time < time_windows[route[i+1]][0]:
                # add on the waiting time
                next_time = time_windows[route[i+1]][0] - time
                futile = False
            elif time+next_time > time_windows[route[i+1]][0]:
                # skip i+1 job
                futile = True
            else:
                futile = False
        else:
            futile = True
        return next_distance, next_time, futile

    return h


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

def collect_data(day:int, seed:int, solution:RoutingSolution, distances:list, times:list, futile:np.array, 
                delivered:np.array, arrival_days:list, time_windows:dict):
    data = {}
    key = 'day'+str(day)+'_'+str(seed)

    time_delivered = []
    for d in delivered:
        if d != 0:
            found = False
            for i,route in enumerate(solution.routes):
                for j,node in enumerate(route):
                    if node == d:
                        time_delivered.append(times[i][j])
                        found = True
                        break
                if found:
                    break
            if not found:
                time_delivered.append(-1) # Error. Hopefully negative time is obviously wrong.

    data[key] = {
        "number_of_packages": int(len(arrival_days)-1), # This could potentially be higher than the number of deliveries in the routes
        "number_of_vehicles": int(len(solution.routes)), # is it number of used vehicles or just a total number? ## Either way, I think we don't actually need this; we can get it from other info
        "distances": [int(sum(veh_dists)) for veh_dists in distances],
        "times": [int(sum(veh_times)) for veh_times in times],
        "deliveries_attempted": [int(len(route)-2) for route in solution.routes],#{[1,3,6,0,0]}, # successful deliveries or total deliveries? ## Attempted, so total. -2 for depo at start and end.
        "futile_deliveries": [int(f) for f in futile],
        "delivered_packages": {
            "days_taken": [int(day - arrival_days[i]) for i in delivered if i != 0], #[1,4,2,7,32,2],
            "time_delivered": [int(t) for t in time_delivered],
            "time_window": [[int(time_windows[i][0]),int(time_windows[i][1])] for i in delivered if i != 0] #[[2,4],[2,3],[2,3],[1,4],[1,2]] 
        },
        "undelivered_packages": {
            "days_taken": [int(day - arrival_days[i]) for i in range(len(arrival_days)) if i!=0 or i not in delivered],#[12,54,21,43,21],
            "time_window": [[int(time_windows[i][0]),int(time_windows[i][1])] for i in range(len(arrival_days)) if i!=0 or i not in delivered]#[[3,7],[2,7],[5,9],[1,3],[4,5]] 
        }
    }

    return data
