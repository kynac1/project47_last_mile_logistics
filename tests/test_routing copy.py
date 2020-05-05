from project47.model import Location, Request
from project47.routing import TimeWindows, BaseProblem
from project47.simulation import *
import numpy as np

def test_base():
    np.random.seed(0)
    locs = [
        Location(172.7319266,	-43.5111688),
        Location(172.63913,	-43.5499101),
        Location(172.6473834,	-43.547876),
        Location(172.5866324,	-43.5295274),
        Location(172.622618,	-43.549314),
        Location(172.6443871,	-43.5194318),
        Location(172.5677181,	-43.5262727),
        Location(172.5768752,	-43.5251111),
        Location(172.6454779,	-43.5308677),
        Location(172.6100037,	-43.526729),
        Location(172.6601199,	-43.5334894),
        Location(172.6593844,	-43.5345317),
        Location(172.7308814,	-43.5199411),
        Location(172.6314508,	-43.5280717),
        Location(172.6215047,	-43.5217694),
        Location(172.7624258,	-43.5761387),
        Location(172.6280757,	-43.5220461),
        Location(172.5798551,	-43.4779506),
        Location(172.619338,	-43.5055327),
        Location(172.6468588,	-43.5255113),
        Location(172.6531345,	-43.5296929),
        Location(172.5716451,	-43.5163431),
        Location(172.7070804,	-43.4955977),
        Location(172.6100037,	-43.526729),
        Location(172.6344832,	-43.5589434),
        Location(172.6365762,	-43.5055445),
        Location(172.6462857,	-43.5170442),
        Location(172.6293588,	-43.5252464),
        Location(172.6486995,	-43.5184613),
        Location(172.6443871,	-43.5194318),
        Location(172.7122268,	-43.4938774),
        Location(172.7462063,	-43.5708894),
        Location(172.5945826,	-43.5132977),
        Location(172.6514435,	-43.5328785),
        Location(172.5464753,	-43.5232292),
        Location(172.5654115,	-43.552104),
        Location(172.5847619,	-43.5278681),
        Location(172.5387021,	-43.5141478),
        Location(172.642015,	-43.5509728),
        Location(172.6234537,	-43.5208447),
        Location(172.6609946,	-43.5679401),
        Location(172.663747,	-43.537706),
        Location(172.6056399,	-43.5374932),
        Location(172.6636662,	-43.5427405),
        Location(172.6303761,	-43.5159515)
    ]

    depo = Location(172.7319266,	-43.5111688)

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