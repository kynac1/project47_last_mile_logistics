from project47.routing import *
import numpy as np


def test_ortools():
    locs = 3
    depo = 0
    distances = np.array([[0, 2, 2], [2, 0, 4], [2, 4, 0]])
    r = ORToolsRouting(locs, 3, depo)
    dim, ind = r.add_dimension(distances, 0, 10, True, "distance")
    r.routing.SetArcCostEvaluatorOfAllVehicles(ind)
    s = r.solve()
    assert r.objective == 8

    # s.plot()

    locs = 3
    depo = 0
    distances = np.array([[0, 2, 2], [2, 0, 4], [2, 4, 0]])
    r = ORToolsRouting(locs, 3, depo)
    dim, ind = r.add_dimension(distances, 0, 10, True, "distance")
    r.routing.SetArcCostEvaluatorOfAllVehicles(ind)
    s = r.solve()
    assert r.objective == 8
    print(s)

    # s.plot()


def test_time_windows():
    locs = 5
    depo = 0
    times = np.array(
        [
            [0, 1, 2, 3, 4],
            [1, 0, 2, 3, 4],
            [2, 2, 0, 3, 4],
            [3, 3, 3, 0, 4],
            [4, 4, 4, 4, 0],
        ]
    )
    windows = np.array([[0.0, 10000.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    r = ORToolsRouting(locs, 3, depo)
    dim, ind = r.add_time_windows(times, windows, 1, 10, False, "time")
    r.routing.SetArcCostEvaluatorOfAllVehicles(ind)
    s = r.solve()
    assert r.objective == 19
    print(s)


def test_disjunction_single():
    locs = 3
    depo = 0
    distances = np.array([[0, 2, 2], [2, 0, 4], [2, 4, 0]])
    r = ORToolsRouting(locs, 3, depo)
    dim, ind = r.add_dimension(distances, 0, 2, True, "distance")
    r.routing.SetArcCostEvaluatorOfAllVehicles(ind)
    for i in range(3):
        r.add_disjunction(i, 10)
    s = r.solve()
    assert r.objective == 20

    r = ORToolsRouting(locs, 1, depo)
    dim, ind = r.add_dimension(distances, 0, 4, True, "distance")
    r.routing.SetArcCostEvaluatorOfAllVehicles(ind)
    for i in range(3):
        r.add_disjunction(i, 10)
    s = r.solve()
    assert r.objective == 14


def test_disjunction_multiple():
    locs = 3
    depo = 0
    distances = np.array([[0, 2, 2], [2, 0, 4], [2, 4, 0]])
    r = ORToolsRouting(locs, 3, depo)
    dim, ind = r.add_dimension(distances, 0, 4, True, "distance")
    r.routing.SetArcCostEvaluatorOfAllVehicles(ind)
    r.add_option([1, 2], 1)
    s = r.solve()
    assert r.objective == 1

    r = ORToolsRouting(locs, 3, depo)
    dim, ind = r.add_dimension(distances, 0, 4, True, "distance")
    r.routing.SetArcCostEvaluatorOfAllVehicles(ind)
    r.add_option([1, 2], 10)
    s = r.solve()
    assert r.objective == 4

    r = ORToolsRouting(locs, 3, depo)
    dim, ind = r.add_dimension(distances, 0, 10, True, "distance")
    r.routing.SetArcCostEvaluatorOfAllVehicles(ind)
    r.add_option([1, 2], 10)
    s = r.solve()
    assert r.objective == 4


def test_two_disjunctions():
    locs = 5
    depo = 0
    distances = np.array(
        [
            [0, 1, 2, 3, 4],
            [1, 0, 2, 3, 4],
            [2, 2, 0, 3, 4],
            [3, 3, 3, 0, 4],
            [4, 4, 4, 4, 0],
        ]
    )
    r = ORToolsRouting(locs, 1, depo)
    dim, ind = r.add_dimension(distances, 0, 100, True, "distance")
    r.routing.SetArcCostEvaluatorOfAllVehicles(ind)
    r.add_option([1, 4], 10)
    r.add_disjunction(4, 10)
    s = r.solve()
    assert r.objective == 13

    r = ORToolsRouting(locs, 1, depo)
    dim, ind = r.add_dimension(distances, 0, 100, True, "distance")
    r.routing.SetArcCostEvaluatorOfAllVehicles(ind)
    r.add_option([1, 4], 10)
    r.add_disjunction(4, 10)
    s = r.solve()
    assert r.objective == 13


if __name__ == "__main__":
    print("sf")
    test_two_disjunctions()
