from project47.data1 import *
from project47.routing import *
from project47.simulation import *
from functools import reduce

def test_ortools_with_osrm():
    """ This test will only pass if the server is running
    """

    # Load in data
    cd = os.path.dirname(os.path.abspath(__file__)).strip('tests') + 'data'
    sample_data = os.path.join(cd,'Toll_CHC_November_Sample_Data.csv')
    CHC_data = os.path.join(cd,'christchurch_street.csv')
    sample_df, CHC_df, CHC_sub, CHC_sub_dict = read_data(sample_data, CHC_data)
    lat, lon = get_sample(5, 0, cd, sample_df, CHC_df, CHC_sub, CHC_sub_dict, save=False)
    dm,tm = osrm_get_dist('', '', lat, lon, host='0.0.0.0:5000', save=False)
    dm = np.array(dm)
    tm = np.array(tm)

    # Randomly assign delivery windows
    delivery_time_windows = {}
    for i in range(1, len(lat)):
        if np.random.rand() > 0.5:
            delivery_time_windows[i] = [0,14400]
        else:
            delivery_time_windows[i] = [14400,28800]

    # Solve routing problem
    locs = dm.shape[0]    
    r = ORToolsRouting(locs, 3)
    dim,ind = r.add_dimension(dm, 0, 50000, True, 'distance')
    r.routing.SetArcCostEvaluatorOfAllVehicles(ind)
    dim,ind = r.add_time_windows(tm, delivery_time_windows, 28800, 28800, True, 'time')
    for i in range(1,locs):
        r.add_disjunction(i,50000)
    r.search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    r.search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    s = r.solve(tlim=10,log=False)
    print(s)

    sim(
        s,
        default_update_function(dm, tm ,delivery_time_windows),
    )

def test_orsm_multiday():
    """ This test will only pass if the server is running

    Also, pretty sure the time windows are bugged in this. They're getting overwritten, and are never removed.
    It's fine for testing other stuff, but we shouldn't copy this behaviour.
    """
    # Setup input files
    cd = os.path.dirname(os.path.abspath(__file__)).strip('tests') + 'data'
    sample_data = os.path.join(cd,'Toll_CHC_November_Sample_Data.csv')
    CHC_data = os.path.join(cd,'christchurch_street.csv')
    sample_df, CHC_df, CHC_sub, CHC_sub_dict = read_data(sample_data, CHC_data)
    
    delivery_lats = np.array([-43.5111688])
    delivery_lons = np.array([172.7319266])

    data = []

    delivery_time_windows = {}
    for day in range(100):

        # Generate new packages and distance matrix
        lat, lon = get_sample(5, 0, cd, sample_df, CHC_df, CHC_sub, CHC_sub_dict, save=False)
        for i in range(max(len(delivery_lats),1), len(delivery_lats)+len(lat)):
            if np.random.rand() > 0.5:
                delivery_time_windows[i] = [0,14400]
            else:
                delivery_time_windows[i] = [14400,28800]

        delivery_lats = np.append(delivery_lats,lat)
        delivery_lons = np.append(delivery_lons,lon)
        dm,tm = osrm_get_dist('', '', delivery_lats, delivery_lons, host='0.0.0.0:5000', save=False)
        if dm is None:
            # We've exceeded the map bounds. Stop here for now, but we should really handle this more gracefully.
            break
        dm = np.array(dm)
        tm = np.array(tm)

        # Solve routing problem
        locs = dm.shape[0]
        r = ORToolsRouting(locs, 3)
        dim,ind = r.add_dimension(dm, 0, 50000, True, 'distance')
        r.routing.SetArcCostEvaluatorOfAllVehicles(ind)
        dim,ind = r.add_time_windows(tm, delivery_time_windows, 28800, 28800, True, 'time')
        for i in range(1,locs):
            r.add_disjunction(i,50000)
        
        #r.search_parameters.first_solution_strategy = (
        #    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        #)
        #r.search_parameters.local_search_metaheuristic = (
        #    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        #)
        s = r.solve(log=False)

        distances, times, futile, delivered = sim(
            s,
            default_update_function(dm, tm , delivery_time_windows),
        )

        undelivered = np.ones(locs, dtype=bool)
        for d in delivered:
            if d != 0:
                undelivered[d] = False

        data.append(collect_data(day, 0, s, distances, times, futile, delivered, [0 for i in range(len(delivery_lats))], delivery_time_windows))
        
        delivery_lats = delivery_lats[undelivered]
        delivery_lons = delivery_lons[undelivered]
        print(len(delivery_lats))


    with open('test_orsm.json', 'w') as outfile:
        json.dump(data, outfile, indent=4)

def test_new_multiday():
    cd = os.path.dirname(os.path.abspath(__file__)).strip('tests') + 'data'
    sample_data = os.path.join(cd,'Toll_CHC_November_Sample_Data.csv')
    CHC_data = os.path.join(cd,'christchurch_street.csv')
    sample_df, CHC_df, CHC_sub, CHC_sub_dict = read_data(sample_data, CHC_data)
    def sample_generator():
        lat, lon = get_sample(5, 0, cd, sample_df, CHC_df, CHC_sub, CHC_sub_dict, save=False)
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
        
        #r.search_parameters.first_solution_strategy = (
        #    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        #)
        #r.search_parameters.local_search_metaheuristic = (
        #    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        #)
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
            default_update_function3(dm, tm, delivery_time_windows)
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

    with open('test_new.json', 'w') as outfile:
        json.dump(data, outfile, indent=4)

if __name__ == "__main__":
    test_new_multiday()