
from project47.routing import *
import numpy as np

def sim(s:RoutingSolution, distance_function, time_function, futile_function, seed:int=0):
    np.random.seed(seed)
    times = np.zeros(len(s.routes))
    distances = np.zeros(len(s.routes))
    futile = np.zeros(len(s.routes))
    delivered = []

    for i,route in enumerate(s.routes):
        for j in range(len(route)-1):
            distances[i] += distance_function(route[j], route[j+1], times[i])
            times[i] += time_function(route[j], route[j+1], times[i])
            f = futile_function(route[j], route[j+1], times[i])
            if f:
                futile[i] += 1
            else:
                delivered.append(route[j])
    
    return distances, times, futile, delivered

def default_distance_function(distance_matrix):
    def f(i,j,time):
        return distance_matrix[i,j]
    return f

def default_time_function(time_matrix):
    def f(i,j,time):
        return time_matrix[i,j]
    return f

def default_futile_function(prob):
    def f(i,j,time):
        return np.random.rand() < prob
    return f