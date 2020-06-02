from project47.model import Location, Request
from project47.routing import *
from project47.simulation import *
import numpy as np

def test_ortools():
    locs = 3
    depo = 0
    distances = np.array([
        [0,2,2],
        [2,0,4],
        [2,4,0]
    ])
    r = ORToolsRouting(locs, 3, depo)
    dim,ind = r.add_dimension(distances, 0, 10, True, 'distance')
    r.routing.SetArcCostEvaluatorOfAllVehicles(ind)
    s = r.solve()
    assert r.objective == 8

    #s.plot()

    locs = 3
    depo = 0
    distances = np.array([
        [0,2,2],
        [2,0,4],
        [2,4,0]
    ])
    r = ORToolsRouting(locs, 3, depo)
    dim,ind = r.add_dimension(distances, 0, 10, True, 'distance')
    r.routing.SetArcCostEvaluatorOfAllVehicles(ind)
    s = r.solve()
    assert r.objective == 8
    print(s)

    #s.plot()

def test_sim_time_windows():
    locs = 5
    depo = 0
    times = np.array([
        [0,1,2,3,4],
        [1,0,2,3,4],
        [2,2,0,3,4],
        [3,3,3,0,4],
        [4,4,4,4,0]
    ])
    windows = np.array([
        [0.,10000.],
        [1.,2.],
        [2.,3.],
        [3.,4.],
        [4.,5.]
    ])

    r = ORToolsRouting(locs, 3, depo)
    dim,ind = r.add_time_windows(times, windows, 1, 10, False, 'time')
    r.routing.SetArcCostEvaluatorOfAllVehicles(ind)
    s = r.solve()
    assert r.objective == 19
    print(s)

    distances = np.array([
        [0,1,2,3,4],
        [1,0,2,3,4],
        [2,2,0,3,4],
        [3,3,3,0,4],
        [4,4,4,4,0]
    ])
    policies=[]
    distance, time, futile, delivered = sim(
        s, 
        default_distance_function(np.zeros((5,5))), 
        default_time_function(times), 
        default_futile_function(0.0),
        windows,
        policies,
        0
    )

    assert all(distance) == 0
    assert max(time == 8)
    assert all(futile == 0)

    # distance, time, futile, delivered = sim(
    #     s, 
    #     default_distance_function(distances), 
    #     default_time_function(np.zeros((3,3))), 
    #     default_futile_function(1)
    # )
    # assert max(distance) == 8
    # assert all(time == 0)
    # assert max(futile == 3)

    #s.plot(times)

if __name__ == "__main__":
    test_sim_time_windows()