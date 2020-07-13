from project47.routing import *
from project47.simulation import *
import numpy as np

def test_reroute_sim_tw():
    times = np.array([
        [0,1,2,3],
        [1,0,2,3],
        [2,2,0,3],
        [3,3,3,0],
    ])
    windows = np.array([
        [0.,10000.],
        [0.,10000.],
        [0.,10000.],
        [2.,3.]
    ])
    
    locs = times.shape[0] 
    depo = 0

    r = ORToolsRouting(locs, 1, depo)
    dim,ind = r.add_time_windows(times, windows, 1, 10, False, 'time')
    r.routing.SetArcCostEvaluatorOfAllVehicles(ind)
    s = r.solve()
    assert r.objective == 9

    distances = np.array([
        [0,1,2,3],
        [1,0,2,3],
        [2,2,0,3],
        [3,3,3,0],
    ])

    windows = np.array([
        [0.,10000.],
        [0.,10000.],
        [0.,10000.],
        [0.,1.]
    ])
    distance, time, futile, delivered = sim(
        s, 
        default_update_function(np.zeros((5,5)), times, windows)
    )

    assert sum(sum(d) for d in distance) == 0
    assert max(max(t) for t in time) == 8
    assert all(futile == 0)

    distance, time, futile, delivered = sim(
        s, 
        update_function4(distances, times, windows)
    )

def test_rerouting_tw():
    
    times = np.array([
        [0,1,2,3],
        [1,0,2,3],
        [2,2,0,3],
        [3,3,3,0],
    ])
    windows = np.array([
        [0.,10000.],
        [0.,10000.],
        [0.,10000.],
        [2.,3.]
    ])
    i = 0
    route = [0, 3, 2, 1, 0]
    places_to_visit = sorted(route[i+2:-1],reverse=False) 
    # places_to_visit = [0, 2, 3]

    # slice the times for the places to visit
    tm = times[places_to_visit]
    tm = tm[:, places_to_visit]
    # slice the time windows for the places to visit
    w = windows[places_to_visit]

    print(tm)

    # if route[i+2] == 0:
    #     next_distance = f(route[i],route[i+2],time)
    #     next_time = g(route[i],route[i+2],time)
    
    # compute rerouting time windows
    w = np.vstack ((w, np.array([0.,99999999999999.])) )
    print ("windows", str(w)) 
    # compute rerouting time matrix
    tm = rerouting_matrix(route[i], tm)
    # printing result 
    print ("times", str(times)) 
    locs = times.shape[0] 
    depo = times.shape[0] - 1


    r = ORToolsRouting(locs, 1, depo)
    dim,ind = r.add_time_windows(times, windows, 1, 10, False, 'time')
    r.routing.SetArcCostEvaluatorOfAllVehicles(ind)
    s = r.solve()
    assert r.objective == 8

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

    route_new = rerouting(k, np.zeros((len(tm),len(tm))), tm, w)
    # route_n = places_to_visit_dic[route_new]
    route_n = [places_to_visit_dic[x] for x in route_new]
    print(route_n)
    
    l = 0

if __name__ == "__main__":
    test_rerouting_tw2()
    test_rerouting_tw()
    test_reroute_sim_tw()