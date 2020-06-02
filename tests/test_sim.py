
from project47.routing import *
from project47.simulation import *

def test_simple_sim():
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

    #s.plot(distances)

    distance, time, futile, delivered = sim(
        s, 
        default_distance_function(distances), 
        default_time_function(np.zeros((3,3))), 
        default_futile_function(0.0)
    )

    assert max(distance) == 8
    assert all(time == 0)
    assert all(futile == 0)

    distance, time, futile, delivered = sim(
        s, 
        default_distance_function(distances), 
        default_time_function(np.zeros((3,3))), 
        default_futile_function(1)
    )
    assert max(distance) == 8
    assert all(time == 0)
    assert max(futile == 3)
