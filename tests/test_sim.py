from project47.routing import *
from project47.simulation import *


def test_simple_sim():
    locs = 3
    depo = 0
    distances = np.array([[0, 2, 2], [2, 0, 4], [2, 4, 0]])
    r = ORToolsRouting(locs, 3, depo)
    dim, ind = r.add_dimension(distances, 0, 10, True, "distance")
    r.routing.SetArcCostEvaluatorOfAllVehicles(ind)
    s = r.solve()
    assert r.objective == 8

    # s.plot(distances)

    distance, time, futile, delivered = sim(
        s, default_update_function(distances, np.zeros((3, 3)), {})
    )

    assert max(max(d) for d in distance) == 8
    assert sum(sum(t) for t in time) == 0
    assert np.allclose(futile, [1, 1, 3])
    assert delivered == []


if __name__ == "__main__":
    test_simple_sim()
