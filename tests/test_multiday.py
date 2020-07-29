from project47.data import *
from project47.routing import *
from project47.simulation import *
from project47.multiday_simulation import *
from functools import reduce

def test_new_multiday():
    """ This is the main example of all the functionality.

    The idea is that when we create a new experiment to run, we'd copy the structure of this function and replace
    parts so that it implements the new policies
    """
    cd = os.path.dirname(os.path.abspath(__file__)).strip('tests') + 'data'
    sample_data = os.path.join(cd,'Toll_CHC_November_Sample_Data.csv')
    CHC_data = os.path.join(cd,'christchurch_street.csv')
    sample_df, CHC_df, CHC_sub, CHC_sub_dict = read_data(sample_data, CHC_data)
    def sample_generator(rg:np.random.Generator):
        lat, lon = get_sample(5, rg, cd, sample_df, CHC_df, CHC_sub, CHC_sub_dict, save=False)
        time_windows = np.zeros((len(lat),2))
        for i in range(len(lat)):
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

        s = r.solve(log=False)
        
        unscheduled = []
        scheduled = reduce(lambda x,y: x+y, s.routes)
        for i in range(locs):
            if i not in scheduled:
                unscheduled.append(i)
        return s, unscheduled
    
    def simulator(routes, dm, tm, delivery_time_windows, rg:np.random.Generator):
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

def test_reproducible():
    """ Run two identical simulations with same random seed. Check they return the same results.
    """
    cd = os.path.dirname(os.path.abspath(__file__)).strip('tests') + 'data'
    sample_data = os.path.join(cd,'Toll_CHC_November_Sample_Data.csv')
    CHC_data = os.path.join(cd,'christchurch_street.csv')
    sample_df, CHC_df, CHC_sub, CHC_sub_dict = read_data(sample_data, CHC_data)
    def sample_generator(rg:np.random.Generator):
        lat, lon = get_sample(5, rg, cd, sample_df, CHC_df, CHC_sub, CHC_sub_dict, save=False)
        time_windows = np.zeros((len(lat),2))
        for i in range(len(lat)):
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

        s = r.solve(log=False)
        
        unscheduled = []
        scheduled = reduce(lambda x,y: x+y, s.routes)
        for i in range(locs):
            if i not in scheduled:
                unscheduled.append(i)
        return s, unscheduled
    
    def simulator(routes, dm, tm, delivery_time_windows, rg:np.random.Generator):
        return sim(
            routes,
            default_update_function3(dm, tm, delivery_time_windows)
        )
    
    data1 = multiday(
        np.array([[-43.5111688],[172.7319266]]),
        sample_generator,
        dist_and_time,
        route_optimizer,
        simulator,
        10,
        0,
        28800,
        seed=123456789
    )

    data2 = multiday(
        np.array([[-43.5111688],[172.7319266]]),
        sample_generator,
        dist_and_time,
        route_optimizer,
        simulator,
        10,
        0,
        28800,
        seed=123456789
    )

    assert data1 == data2