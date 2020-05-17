from project47.model import Location, Request
from project47.routing import *
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
    r.solve()
    assert r.objective == 8

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

def test_time_windows():
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

if __name__ == "__main__":
    test_time_windows()