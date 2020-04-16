from project47.model import Location, Request
from project47.routing import TimeWindows, BaseProblem
from project47.simulation import *
import numpy as np

def test_base():
    np.random.seed(0)
    locs = [
        Location(np.random.rand(), np.random.rand()) for _ in range(50)
    ]

    depo = Location(np.random.rand(), np.random.rand())

    prob = BaseProblem(depo, locs, 100)
    def distance(a,b):
        return a.distance_to(b)

    dist_dim, dist_ind = prob.add_dimension(distance, 0, 300000, True, "Distance")
    #dist_dim.SetGlobalSpanCostCoefficient(100)
    prob.routing.SetArcCostEvaluatorOfAllVehicles(dist_ind)

    solution = None
    for i in range(3):
        prob.search_parameters.first_solution_strategy = (
            14
        )

        prob.search_parameters.local_search_metaheuristic = (
            i
        )
        new = prob.solve(log=True,tlim=30)
        if new:
            solution = new

    if solution:
        print(str(solution))
        print(static_sim(solution))
        solution.plot()

def test_time_windows():
    reqs = [
        Request(np.random.rand(), np.random.rand(), round(np.random.rand()*10), 10+round(np.random.rand())*10) for _ in range(1)
    ]

    depo = Location(np.random.rand(), np.random.rand())

    prob = TimeWindows(depo, reqs, 5)
    def distance(a,b):
        a.distance_to(b)
    def time(a,b):
        a.time_to(b)

    dist_dim,_ = prob.add_dimension(distance, 0, 100000000, True, "Distance")
    dist_dim.SetGlobalSpanCostCoefficient(100)
    prob.add_time_windows(time, 0, 23000000, False, "Time")

    solution = prob.solve(log=True)
    if solution:
        print(str(solution))
        solution.plot()

if __name__ == "__main__":
    test_base()