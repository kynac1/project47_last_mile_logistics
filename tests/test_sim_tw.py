from project47.model import Location, Request
from project47.routing import *
from project47.simulation import *
import numpy as np

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
    distance, time, futile, delivered = sim(
        s, 
        default_update_function(np.zeros((5,5)), times, windows)
    )

    assert all(distance) == 0
    assert max(time == 8)
    assert all(futile == 0)

    distance, time, futile, delivered = sim(
        s, 
        update_function2(distances, times, windows)
    )

if __name__ == "__main__":
    test_sim_time_windows()