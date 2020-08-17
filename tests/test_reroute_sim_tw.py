import numpy as np
from project47.routing import *
from project47.simulation import *
from tests.test_customer_tw import *
from project47.customer import Customer
from functools import reduce
from numpy.random import Generator, PCG64


def test_reroute_sim_tw():
    
    times = np.array([
        [0,1,2,3,1],
        [1,0,2,3,1],
        [2,2,0,3,4],
        [3,3,3,0,4],
        [1,1,4,4,0]
    ])

    times = np.array([
        [0,1,2,3,1],
        [1,0,2,9,1],
        [2,2,0,3,4],
        [3,9,3,0,4],
        [1,1,4,4,0]
    ])

    windows = np.array([
        [0.,10000.],
        [0.,10000.],
        [0.,10000.],
        [2.,3.],
        [0.,10000.]
    ])

    distances = np.array([
        [0,1,2,3,1],
        [1,0,2,9,1],
        [2,2,0,3,4],
        [3,9,3,0,4],
        [1,1,4,4,0]
    ])
    
    depo = 0
    locs = times.shape[0]   
    r = ORToolsRouting(locs, 3, depo)
    dim,ind = r.add_time_windows(times, windows, 1, 10, False, 'time')
    r.routing.SetArcCostEvaluatorOfAllVehicles(ind)
    s = r.solve()
    s.routes = [[0, 3, 2, 1, 4, 0]]
    print(s)

    # seed=123456789
    # rg = Generator(PCG64(seed))
    # tw, customers = sample_generator(rg, 3)

    windows = np.array([
        [0.,10000.],
        [0.,10000.],
        [0.,1.],
        [2.,3.],
        [0.,10000.]
    ])

    # windows = np.array([
    #     [0.,10000.],
    #     [0.,10000.],
    #     [0.,10000.],
    #     [0.,1.],
    #     [0.,10000.]
    # ])

    distance, time, futile, delivered = sim1(
        s, 
        update_function4(distances, times, windows)#, customers)
    )

    # skip place 2
    i = 1
    routes = [[0, 3, 2, 1, 4, 0]]

# no policy
def update_function10(distance_matrix, time_matrix, time_windows, customers):
    '''
    This time window policy makes the decision after arriving at the next place.
    The deliver man checks the time once arrived. 
    If the current time falls out of the time windows, then he will skip and go to the next place.
    '''
    f = default_distance_function(distance_matrix)
    g = default_time_function(time_matrix)
    # c = default_customers(time_matrix)
    def h(route, i, time):
        interval_presence = 3600
        next_distance = f(route[i],route[i+1],time)
        next_time = g(route[i],route[i+1],time)
        indp = int( (time+next_time)/interval_presence )
        futile = not bool(customers[i].presence[indp]) # check if the next delivery is going to be futile
        return next_distance, next_time, futile, route
    return h

# estimate ahead
def update_function20(distance_matrix, time_matrix, time_windows, customers):
    '''
    This time window policy makes the decision after arriving at the next place.
    The deliver man checks the time once arrived. 
    If the current time falls out of the time windows, then he will skip and go to the next place.
    '''
    f = default_distance_function(distance_matrix)
    g = default_time_function(time_matrix)
    # c = default_customers(time_matrix)
    def h(route, i, time):
        interval_presence = 3600
        next_distance = f(route[i],route[i+1],time)
        next_time = g(route[i],route[i+1],time)
        if route[i+1] in time_windows: # not sure if this line is needed as all loc has a tw?
            # out of time window
            if time+next_time < time_windows[route[i+1]][0] or time+next_time > time_windows[route[i+1]][1]:
                # go straight to depot if the next place is depot after skipping
                if route[i+2] == 0:
                    next_distance = f(route[i],route[i+2],time)
                    next_time = g(route[i],route[i+2],time)
                    route = [0]
                    futile = True # manually set to Ture to avoid appending depot to the delivery list in the simulation
                    return next_distance, next_time, futile, route
                # skip i+1 job and reroute
                else:
                    route = rerouting(i, route, distance_matrix, time_matrix, time_windows)
                    print(route)
                    next_distance = f(route[0],route[1],time)
                    next_time = g(route[0],route[1],time)
        # index of presence according to next arrival time
        indp = int( (time+next_time)/interval_presence )
        futile = not bool(customers[i].presence[indp])

        return next_distance, next_time, futile, route
    return h


# calling policy
def update_function30(distance_matrix, time_matrix, time_windows, customers):
    '''
    This time window policy makes the decision after arriving at the next place.
    The deliver man checks the time once arrived. 
    If the current time falls out of the time windows, then he will skip and go to the next place.
    '''
    f = default_distance_function(distance_matrix)
    g = default_time_function(time_matrix)
    # c = default_customers(time_matrix)
    def h(route, i, time):
        interval_presence = 3600
        next_distance = f(route[i],route[i+1],time)
        next_time = g(route[i],route[i+1],time)
        if rg.random() > 0.5: # if the tw is changed after the call
            time_windows[route[i+1]]= random_time_window_generator(rg)
        
        if route[i+1] in time_windows: # not sure if this line is needed as all loc has a tw?
            # out of time window
            if time+next_time < time_windows[route[i+1]][0] or time+next_time > time_windows[route[i+1]][1]:
                # go straight to depot if the next place is depot after skipping
                if route[i+2] == 0:
                    next_distance = f(route[i],route[i+2],time)
                    next_time = g(route[i],route[i+2],time)
                    route = [0]
                    futile = True # manually set to Ture to avoid appending depot to the delivery list in the simulation
                    return next_distance, next_time, futile, route
                # skip i+1 job and reroute
                else:
                    route = rerouting(i, route, distance_matrix, time_matrix, time_windows)
                    print(route)
                    next_distance = f(route[0],route[1],time)
                    next_time = g(route[0],route[1],time)
        # index of presence according to next arrival time
        indp = int( (time+next_time)/interval_presence )
        futile = not bool(customers[i].presence[indp])

        return next_distance, next_time, futile, route
    return h

def sim1(s:RoutingSolution, update_function):
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
                # delivered.append(route[j])
            
            if isfutile:
                futile[i] += 1
            else:
                delivered.append(route[j+1])
            
            j = j+1
                
    return distances, times, futile, delivered




def test_rerouting_tw3():
    
    times = np.array([
        [0,1,2,3,1],
        [1,0,2,3,1],
        [2,2,0,3,4],
        [3,3,3,0,4],
        [1,1,4,4,0]
    ])

    times = np.array([
        [0,1,2,3,1],
        [1,0,2,9,1],
        [2,2,0,3,4],
        [3,9,3,0,4],
        [1,1,4,4,0]
    ])

    windows = np.array([
        [0.,10000.],
        [0.,10000.],
        [0.,10000.],
        [2.,3.],
        [0.,10000.]
    ])
    # skip place 2
    i = 1
    route = [0, 3, 2, 1, 4, 0]
    
    route_new = rerouting(i, route, np.zeros((len(times),len(times))), times, windows)
    print(route_new)

    l = 0


if __name__ == "__main__":
    test_reroute_sim_tw()
    test_rerouting_tw3()

    