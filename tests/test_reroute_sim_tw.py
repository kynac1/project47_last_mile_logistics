from project47.routing import *
from project47.simulation import *
import numpy as np

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

    windows = np.array([
        [0.,10000.],
        [0.,10000.],
        [0.,1.],
        [2.,3.],
        [0.,10000.]
    ])

    distance, time, futile, delivered = sim1(
        s, 
        update_function4(distances, times, windows)
    )

    # skip place 2
    i = 1
    routes = [[0, 3, 2, 1, 4, 0]]
 
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
                delivered.append(route[j])
            
            if isfutile:
                futile[i] += 1
            else:
                delivered.append(route[j])
            
            j = j+1
                
    return distances, times, futile, delivered

def test_rerouting_tw2():
    
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
    # slice the times for the places to visit
    tm = times[places_to_visit]
    tm = tm[:, places_to_visit]
    # slice the time windows for the places to visit
    w = windows[places_to_visit]
    print(tm)

    route_new = rerouting(k, route, np.zeros((len(tm),len(tm))), tm, w)
    # route_n = places_to_visit_dic[route_new]
    route_n = [places_to_visit_dic[x] for x in route_new]
    print(route_n)

    l = 0

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
    test_rerouting_tw2()
    test_rerouting_tw3()

    