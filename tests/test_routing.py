from project47.model import Location, Request
from project47.routing import TimeWindows, BaseProblem
import numpy as np

def test_base():
    locs = [
        Location(np.random.rand()*10, np.random.rand()*10) for _ in range(10)
    ]

    depo = Location(np.random.rand()*10, np.random.rand()*10)

    prob = BaseProblem(depo, locs, 5)
    prob.solve()

    prob.plot_solution()
    prob.print_solution()

def test_time_windows():
    reqs = [
        Request(np.random.rand(), np.random.rand(), round(np.random.rand()*10), 10+round(np.random.rand())*10) for _ in range(1)
    ]

    depo = Location(np.random.rand(), np.random.rand())

    prob = TimeWindows(depo, reqs, 5)
    prob.solve()

    prob.plot_solution()
    prob.print_solution()

if __name__ == "__main__":
    test_base()