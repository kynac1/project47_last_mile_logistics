from project47.simulation import *
from project47.routing import *
import numpy as np
import time

logging.basicConfig(level="ERROR")


def generate_problem(n: int, rg: np.random.Generator):
    route = list(range(n))
    rg.shuffle(route)
    dm = rg.random((n, n)) * 1000
    tm = rg.random((n, n)) * 1000
    tw = rg.random(n) * 1000
    tw = np.vstack((tw, tw + 1000)).T
    t = rg.random() * 100
    return [route, dm, tm, tw, t, {}]


def benchmark_strategy(problem, tlim: int, fss, lsm):
    res = []
    kwargs = {
        "tlim": tlim,
        "fss": fss,
        "lsm": lsm,
        "return_obj": True,
    }
    start = time.time()
    res = rerouting_new(1, *problem, **kwargs)
    end = time.time()
    return res, end - start


def test_a():
    fss_list = [
        routing_enums_pb2.FirstSolutionStrategy.ALL_UNPERFORMED,
        routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES,
        routing_enums_pb2.FirstSolutionStrategy.FIRST_UNBOUND_MIN_VALUE,
        routing_enums_pb2.FirstSolutionStrategy.GLOBAL_CHEAPEST_ARC,
        routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_ARC,
        routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION,
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
        routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC,
        routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
        routing_enums_pb2.FirstSolutionStrategy.SEQUENTIAL_CHEAPEST_INSERTION,
    ]
    lsm_list = [
        routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC,
    ]
    """ routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
        routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT,
        routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
    ]"""
    return fss_list, lsm_list, [1, 10, 20]


def test_b():
    rg = np.random.Generator(np.random.PCG64(123456789))
    problems = [generate_problem(100, rg) for _ in range(5)]
    fss_list = [
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
        routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC,
    ]
    lsm_list = [
        routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC,
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
        routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT,
        routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
    ]
    return fss_list, lsm_list, [1, 10, 20]


def test_c():
    rg = np.random.Generator(np.random.PCG64(1234567890))
    problems = [generate_problem(100, rg) for _ in range(5)]
    fss_list = [
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
        routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC,
    ]
    lsm_list = [
        routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
    ]
    return fss_list, lsm_list, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


if __name__ == "__main__":
    fss_list, lsm_list, tlims = test_b()
    rg = np.random.Generator(np.random.PCG64(123456789))
    problems = [generate_problem(100, rg) for _ in range(5)]
    fss_indexer = {
        v: k for k, v in routing_enums_pb2.FirstSolutionStrategy.__dict__.items()
    }
    lsm_indexer = {
        v: k for k, v in routing_enums_pb2.LocalSearchMetaheuristic.__dict__.items()
    }
    for fss in fss_list:
        for lsm in lsm_list:
            for tlim in tlims:
                for i, problem in enumerate(problems):
                    try:
                        res = [
                            *benchmark_strategy(
                                problem,
                                tlim,
                                fss,
                                lsm,
                            ),
                            fss_indexer[fss],
                            lsm_indexer[lsm],
                            tlim,
                            i,
                        ]
                    except Exception as e:
                        logger.error(e)
                        res = [np.inf, 0, fss_indexer[fss], lsm_indexer[lsm], tlim, i]
                    with open(
                        f"ortools_benchmarks/rerouting_results/rerouting3.txt", "a"
                    ) as f:
                        f.write(str(res).strip("[]"))
                        f.write("\n")
                    print(res)
