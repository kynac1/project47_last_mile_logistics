import os
import numpy as np
from project47.simulation import *
from project47.data import *
from project47.routing import *
from project47.multiday_simulation import *
from functools import reduce
from multiprocessing import Process

def no_time_windows_comparison(arrival_rate, num_vehicles, num_time_windows):

    day_start = 0
    day_end = 28800

    cd = os.path.dirname(os.path.abspath(__file__)).strip('experiments') + 'data'
    sample_data = os.path.join(cd,'Toll_CHC_November_Sample_Data.csv')
    CHC_data = os.path.join(cd,'christchurch_street.csv')
    sample_df, CHC_df, CHC_sub, CHC_sub_dict = read_data(sample_data, CHC_data)
    def sample_generator(rg:np.random.Generator):
        lat, lon = get_sample(rg.poisson(arrival_rate), rg, cd, sample_df, CHC_df, CHC_sub, CHC_sub_dict, save=False)
        time_windows = np.zeros((len(lat),2))
        for i in range(len(lat)):
            interval = (day_end - day_start) / num_time_windows
            for j in range(num_time_windows):
                if rg.random() > (num_time_windows - (j+1))/num_time_windows:
                    time_windows[i,0] = interval*j
                    time_windows[i,1] = interval*(j+1)
        return lat, lon, time_windows

    def dist_and_time(lats, lons):
        return osrm_get_dist('', '', lats, lons, host='0.0.0.0:5000', save=False)

    def route_optimizer(depots, dm, tm, time_windows, day, arrival_days, futile_count):
        locs = dm.shape[0]
        r = ORToolsRouting(locs, num_vehicles)
        dim,ind = r.add_dimension(dm, 0, 50000, True, 'distance')
        r.routing.SetArcCostEvaluatorOfAllVehicles(ind)
        dim,ind = r.add_time_windows(tm, time_windows, day_end, day_end, False, 'time')
        for i in range(1,locs):
            r.add_disjunction(i,50000)
        
        r.search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        r.search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        s = r.solve(log=False, tlim=20)
        
        unscheduled = []
        scheduled = reduce(lambda x,y: x+y, s.routes)
        for i in range(locs):
            if i not in scheduled:
                unscheduled.append(i)
        return s, unscheduled
    
    def simulator(routes, dm, tm, delivery_time_windows, rg):
        return sim(
            routes,
            constant_futility_update(dm, tm, delivery_time_windows, 0.1, rg)
        )

    data = multiday(
        np.array([[-43.5111688],[172.7319266]]),
        sample_generator,
        dist_and_time,
        route_optimizer,
        simulator,
        100,
        day_start,
        day_end
    )

    with open(f'experiments/results/experiments_{num_time_windows}_{arrival_rate}_{num_vehicles}.json', 'w') as outfile:
        json.dump(data, outfile, separators=(',',':'))

def constant_futility_update(distance_matrix, time_matrix, time_windows, futile_rate, rg):
    ''' Checks if the time to go to the next location will make them late.
    If so they don't go.
    However, the true futile rate is actually constant, and doesn't vary with time.
    '''
    f = default_distance_function(distance_matrix)
    g = default_time_function(time_matrix)

    def h(route, i, time):
        next_distance = f(route[i],route[i+1],time)
        next_time = g(route[i],route[i+1],time)
        if time + next_time > time_windows[route[i+1]][1]:
            return 0, 0, True
        futile = rg.random.rand() < futile_rate 
        return next_distance, next_time, futile

    return h

if __name__ == "__main__":
    try:
        with open(f'experiments/results/experiments_4_time_windows_20_5.json', 'w') as f:
            pass
        print("OK")
    except:
        print("Cannot open file location")
        exit()
    for rate in [10,20]:
        for vehs in [3,5]:
            print(f"Running {rate}, {vehs}")
            p1 = Process(target=no_time_windows_comparison, args=(rate,vehs,1))
            p2 = Process(target=no_time_windows_comparison, args=(rate,vehs,4))
            p1.start()
            p2.start()
            p1.join()
            p2.join()
            
