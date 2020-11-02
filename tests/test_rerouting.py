from project47.data import *
from project47.routing import *
from project47.simulation import *


def test_rerouting():
    locs = 3
    depo = 0
    distances = np.array([[0, 2, 2], [2, 0, 4], [2, 4, 0]])

    distances = np.array([[0, 1, 2, 3], [1, 0, 2, 3], [2, 2, 0, 3], [3, 3, 3, 0]])

    distances = rerouting_matrix(3, distances)

    # printing result
    print("resultant array", str(distances))
    locs = distances.shape[0]
    depo = distances.shape[0] - 1

    r = ORToolsRouting(locs, 1, depo)
    dim, ind = r.add_dimension(distances, 0, 10, True, "distance")
    r.routing.SetArcCostEvaluatorOfAllVehicles(ind)
    s = r.solve()
    route = s.routes[0][1:-1]
    assert r.objective == 6

    # s.plot()


def test_rerouting_tw():

    times = np.array(
        [
            [0, 1, 2, 3],
            [1, 0, 2, 3],
            [2, 2, 0, 3],
            [3, 3, 3, 0],
        ]
    )
    windows = np.array([[0.0, 10000.0], [0.0, 10000.0], [0.0, 10000.0], [2.0, 3.0]])

    # compute rerouting time windows
    windows = np.vstack((windows, np.array([0.0, 99999999999999.0])))
    print("windows", str(windows))
    # compute rerouting time matrix
    times = rerouting_matrix(3, times)
    # printing result
    print("times", str(times))
    locs = times.shape[0]
    depo = times.shape[0] - 1

    r = ORToolsRouting(locs, 1, depo)
    dim, ind = r.add_time_windows(times, windows, 1, 10, False, "time")
    r.routing.SetArcCostEvaluatorOfAllVehicles(ind)
    s = r.solve()
    assert r.objective == 6


def test_ortools():
    locs = 3
    depo = 0
    distances = np.array([[0, 2, 2], [2, 0, 4], [2, 4, 0]])
    r = ORToolsRouting(locs, 1, depo)
    dim, ind = r.add_dimension(distances, 0, 10, True, "distance")
    r.routing.SetArcCostEvaluatorOfAllVehicles(ind)
    s = r.solve()
    assert r.objective == 8
    print(s)


def test_time_windows():

    times = np.array(
        [
            [0, 1, 2, 3],
            [1, 0, 2, 3],
            [2, 2, 0, 3],
            [3, 3, 3, 0],
        ]
    )
    windows = np.array([[0.0, 10000.0], [0.0, 10000.0], [0.0, 10000.0], [2.0, 3.0]])

    locs = times.shape[0]
    depo = 0

    r = ORToolsRouting(locs, 1, depo)
    dim, ind = r.add_time_windows(times, windows, 1, 10, False, "time")
    r.routing.SetArcCostEvaluatorOfAllVehicles(ind)
    s = r.solve()
    assert r.objective == 9
    print(s)


if __name__ == "__main__":
    # test_ortools()
    test_time_windows()

    test_rerouting()
    # test_rerouting_tw()
