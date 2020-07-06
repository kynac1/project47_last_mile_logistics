import os
import numpy as np
from project47.simulation import *
from project47.data1 import *
from project47.routing import *
from functools import reduce

def no_time_windows(arrival_rate):
    """ Still enforces that jobs are done within working hours, but nothing more.
    """
    cd = os.path.dirname(os.path.abspath(__file__)).strip('experiments') + 'data'
    sample_data = os.path.join(cd,'Toll_CHC_November_Sample_Data.csv')
    CHC_data = os.path.join(cd,'christchurch_street.csv')
    sample_df, CHC_df, CHC_sub, CHC_sub_dict = read_data(sample_data, CHC_data)
    def sample_generator():
        lat, lon = get_sample(np.random.poisson(arrival_rate), None, cd, sample_df, CHC_df, CHC_sub, CHC_sub_dict, save=False)
        time_windows = np.zeros((len(lat),2))
        for i in range(len(lat)):
            if np.random.rand() > 0.5:
                time_windows[i,0] = 0
                time_windows[i,1] = 28800
            else:
                time_windows[i,0] = 0
                time_windows[i,1] = 28800
        return lat, lon, time_windows

    def dist_and_time(lats, lons):
        return osrm_get_dist('', '', lats, lons, host='0.0.0.0:5000', save=False)

    def route_optimizer(depots, dm, tm, time_windows, day, arrival_days, futile_count):
        locs = dm.shape[0]
        r = ORToolsRouting(locs, 5)
        dim,ind = r.add_dimension(dm, 0, 50000, True, 'distance')
        r.routing.SetArcCostEvaluatorOfAllVehicles(ind)
        dim,ind = r.add_time_windows(tm, time_windows, 28800, 28800, True, 'time')
        for i in range(1,locs):
            r.add_disjunction(i,50000)
        
        r.search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        r.search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        s = r.solve(log=False)
        
        unscheduled = []
        scheduled = reduce(lambda x,y: x+y, s.routes)
        for i in range(locs):
            if i not in scheduled:
                unscheduled.append(i)
        return s, unscheduled
    
    def simulator(routes, dm, tm, delivery_time_windows):
        return sim(
            routes,
            default_update_function3(dm, tm, delivery_time_windows),
            None
        )

    data = multiday(
        np.array([[-43.5111688],[172.7319266]]),
        sample_generator,
        dist_and_time,
        route_optimizer,
        simulator,
        10,
        0,
        28800
    )

    with open(f'experiments_no_time_arrival_{arrival_rate}.json', 'w') as outfile:
        json.dump(data, outfile, indent=2)


if __name__ == "__main__":
    no_time_windows(20)
    