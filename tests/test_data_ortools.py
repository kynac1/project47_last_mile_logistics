from project47.data1 import *
from project47.routing import *
from project47.simulation import *

def test_ortools_with_osrm():
    """ This test will only pass if the server is running
    """

    # Load in data
    cd = os.path.dirname(os.path.abspath(__file__)).strip('tests') + 'data'
    sample_data = os.path.join(cd,'Toll_CHC_November_Sample_Data.csv')
    CHC_data = os.path.join(cd,'christchurch_street.csv')
    lat, lon = get_sample(10, 0, '', sample_data, CHC_data, False)
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
    
    delivery_lats = np.array([-43.5111688])
    delivery_lons = np.array([172.7319266])


    data = []

    delivery_time_windows = {}
    for day in range(2):

        # Generate new packages and distance matrix
        lat, lon = get_sample(20, 0, '', sample_data, CHC_data, False)
        for i in range(max(len(delivery_lats),1), len(delivery_lats)+len(lat)):
            if np.random.rand() > 0.5:
                delivery_time_windows[i] = [0,14400]
            else:
                delivery_time_windows[i] = [14400,28800]

        delivery_lats = np.append(delivery_lats,lat)
        delivery_lons = np.append(delivery_lons,lon)
        dm,tm = osrm_get_dist('', '', delivery_lats, delivery_lons, host='0.0.0.0:5000', save=False)
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
            default_update_function(dm, tm ,delivery_time_windows),
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


if __name__ == "__main__":
    test_orsm_multiday()