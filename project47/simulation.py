
from project47.routing import *
import numpy as np

def static_sim(s:Solution, seed:int=0):
    """ Not really a sim at all. Basically just a demo of how to traverse the solution, and produce some results

    There's some amout of randomness added, just for fun.
    """
    np.random.seed(seed)
    times = np.zeros(len(s.vehicles))
    futile = np.zeros(len(s.vehicles), dtype=int)

    def isfutile(l:Location):
        v = np.random.rand()
        return v < 0.1

    for i,route in enumerate(s.routes):
        for j, loc in enumerate(route):
            if j < len(route)-1:
                times[i] += loc.time_to(route[j+1])+np.random.rand()
            else:
                times[i] += loc.time_to(route[0])+np.random.rand()
            futile[i] += isfutile(loc)
    
    return times, futile