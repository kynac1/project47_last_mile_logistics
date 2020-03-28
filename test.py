from project47.model import Location
from project47.routing import BaseProblem
import numpy as np

locs = [
    Location(np.random.rand()*10, np.random.rand()*10) for _ in range(10)
]

depo = Location(np.random.rand()*10, np.random.rand()*10)

prob = BaseProblem(depo, locs, 5)
prob.plot_solution()
prob.solve()

prob.print_solution()