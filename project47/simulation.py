
from project47.routing import *
import numpy as np
from copy import copy
import json
import os

def sim(s:RoutingSolution, update_function):
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
    times = []
    distances = []
    futile = np.zeros(len(s.routes))
    delivered = []

    for i,route in enumerate(s.routes):
        times.append([])
        times[-1].append(0)
        distances.append([])
        distances[-1].append(0)

        j = 0
        while j < (len(route)-1):
            distance, time, isfutile, route_new = update_function(route, j, times[-1][-1])
            distances[-1].append(distances[-1][-1] + distance)
            times[-1].append(times[-1][-1] + time)
            # compare two routes, update the route and the index
            if route != route_new:
                j = 0
                route = route_new
                delivered.append(route[j])
            
            if isfutile:
                futile[i] += 1
            else:
                delivered.append(route[j])
            
            j = j+1

        # for j in range(len(route)-1):
        #     distance, time, isfutile = update_function(route, j, times[-1][-1])
        #     distances[-1].append(distances[-1][-1] + distance)
        #     times[-1].append(times[-1][-1] + time)
        #     if isfutile:
        #         futile[i] += 1
        #     else:
        #         delivered.append(route[j])
                
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
            futile = True # why futile is true if it does not have a tw?
        return next_distance, next_time, futile, route

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
            elif time+next_time > time_windows[route[i+1]][0]: # second index of tw should be 1?
                # skip i+1 job
                futile = True
            else:
                futile = False
        else:
            futile = True
        return next_distance, next_time, futile, route

    return h

def default_update_function3(distance_matrix, time_matrix, time_windows):
    ''' Basically the same as above, but changed the format of time_windows to a np.array
    This time window policy makes the decision after arriving at the next place.
    The deliver man checks the time once arrived. 
    If the current time falls out of the time windows, then he will skip and go to the next place.
    '''
    f = default_distance_function(distance_matrix)
    g = default_time_function(time_matrix)

    def h(route, i, time):
        next_distance = f(route[i],route[i+1],time)
        next_time = g(route[i],route[i+1],time)
        futile = time+next_time < time_windows[route[i+1]][0] or time+next_time > time_windows[route[i+1]][1]
        return next_distance, next_time, futile, route

    return h

def update_function4(distance_matrix, time_matrix, time_windows):
    '''
    This time window policy makes the decision before arriving at the next place.
    The deliver man checks the time before departing from the current place. 
    If the estimated arriving time is earlier than the time windows, then he will wait till the availabe time.
    If the estimated arriving time is later than the time windows, he will skip the next place and reroute.
    '''

    f = default_distance_function(distance_matrix)
    g = default_time_function(time_matrix)
    def h(route, i, time):
        # route = [0, 3, 2, 1, 4, 0]
        next_distance = f(route[i],route[i+1],time)
        next_time = g(route[i],route[i+1],time)
        if time+next_time < time_windows[route[i+1]][0]:
            # add on the waiting time
            next_time = time_windows[route[i+1]][0] - time
            futile = False
        elif time+next_time > time_windows[route[i+1]][1]:
            # skip i+1 job and reroute
            futile = True
            # go straight to depot if the next place is depot after skipping
            if route[i+2] == 0:
                next_distance = f(route[i],route[i+2],time)
                next_time = g(route[i],route[i+2],time)
            else:
                # # get the rest of the places that need to visit
                # places_to_visit = route[i+2::]
                # if i != 0:
                #     places_to_visit.append(route[i])
                # places_to_visit.sort()
                # keys = list(np.arange(len(places_to_visit)))
                # # record both the places that their indices in rerouting
                # places_to_visit_dic = dict(zip(keys, places_to_visit))
                # # current place - starting place for rerouting
                # k = places_to_visit.index(route[i])
                # # slice the times for the places to visit
                # tm = time_matrix[places_to_visit]
                # tm = tm[:, places_to_visit]
                # # slice the time windows for the places to visit
                # w = time_windows[places_to_visit]
                # # print(tm)
                # route_new = rerouting(k, np.zeros((len(tm),len(tm))), tm, w)
                # route_n = [places_to_visit_dic[x] for x in route_new]
                # # print(route_n)
                route = rerouting(i, route, distance_matrix, time_matrix, time_windows)
                print(route)

                next_distance = f(route[0],route[1],time)
                next_time = g(route[0],route[1],time)
        else:
            futile = False
        
        return next_distance, next_time, futile, route

    return h

def update_function5(distance_matrix, time_matrix, time_windows, rg:np.random.Generator):
    '''
    This time window policy makes the decision before arriving at the next place.
    The deliver man calls the next customer before departing from the current place. 
    If the time window has changed, reroute.
    If the time window stays the same, he continues the route
    '''
    f = default_distance_function(distance_matrix)
    g = default_time_function(time_matrix)
    def h(route, i, time):
        # route = [0, 3, 2, 1, 4, 0]
        next_distance = f(route[i],route[i+1],time)
        next_time = g(route[i],route[i+1],time)
        if rg.random() > 0.5: # if the tw is changed after the call
            
            time_windows[route[i+1]]= random_time_window_generator(rg)
        if time+next_time < time_windows[route[i+1]][0]:
            # add on the waiting time
            next_time = time_windows[route[i+1]][0] - time
            futile = False
        elif time+next_time > time_windows[route[i+1]][1]:
            # skip i+1 job and reroute
            futile = True
            # go straight to depot if the next place is depot after skipping
            if route[i+2] == 0:
                next_distance = f(route[i],route[i+2],time)
                next_time = g(route[i],route[i+2],time)
            else:
                route = rerouting(i, route, distance_matrix, time_matrix, time_windows)
                print(route)

                next_distance = f(route[0],route[1],time)
                next_time = g(route[0],route[1],time)
        else:
            futile = False
        
        return next_distance, next_time, futile, route

    return h

def default_distance_function(distance_matrix):
    def f(i,j,time):
        return distance_matrix[i,j]
    return f

def default_time_function(time_matrix):
    def f(i,j,time):
        return time_matrix[i,j]
    return f

def default_futile_function(prob, rg:np.random.Generator):
    def f(i,j,time):
        return rg.random() < prob
    return f

def random_time_window_generator(rg):
    time_windows = np.zeros(2)
    if rg.random() > 0.5:
        time_windows[0] = 0
        time_windows[1] = 28800
    else:
        time_windows[0] = 0
        time_windows[1] = 28800
    return time_windows
    
def rerouting(i, route, distance_matrix, time_matrix, time_windows):
    '''
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
    '''
    # dm = default_distance_function(distance_matrix)
    # tm = default_time_function(time_matrix)

    # if route[i+2] == 0:
    #     next_distance = f(route[i],route[i+2],time)
    #     next_time = g(route[i],route[i+2],time)

    # get the rest of the places that need to visit
    places_to_visit = route[i+2::]
    if i != 0:
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
    tw = np.vstack ((tw, np.array([0.,99999999999999.])) )
    # compute rerouting time matrix
    tm = rerouting_matrix(k, tm)
    # compute rerouting distance matrix
    dm = rerouting_matrix(k, dm)
    # print ("time_matrix", str(tm)) # printing result 
    locs = tm.shape[0] 
    depo = tm.shape[0] - 1

    # solve the problem
    r = ORToolsRouting(locs, 1, depo)
    dim,ind = r.add_time_windows(tm, tw, 1, 10, False, 'time')
    r.routing.SetArcCostEvaluatorOfAllVehicles(ind)
    s = r.solve()
    route_new = s.routes[0][1:-1]

    # route_n = places_to_visit_dic[route_new]
    route_n = [places_to_visit_dic[x] for x in route_new]

    return route_n

def rerouting_matrix(k, matrix):
    '''
    This function computes the dm or tm needed for rerouting.

    Parameters
    ---------------
    k: current starting place
    matrix: original distance matrix or time matrix

    Returns
    ---------------
    matrix: dm or tm with an arbitrary depot 
    '''
    M = 99999999999999
    # add an arbitrary depot 'E'
    row_to_be_added = np.ones(matrix.shape[0]) * M
    # Adding row to numpy array 
    matrix = np.vstack ((matrix, row_to_be_added) ) 
    col_to_be_added = np.ones(matrix.shape[0]) * M
    # Adding col to numpy array 
    matrix = np.hstack ((matrix, np.atleast_2d(col_to_be_added).T) )
    # force an arc from 'E' to current place - to start from 'E'
    matrix[-1][k] = 0
    # force an arc from depot 'O' to 'E' - to arrive at 'O'
    matrix[0][-1] = 0

    return matrix
