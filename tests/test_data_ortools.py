from project47.data1 import *
from project47.routing import *

def test_ortools_with_osrm():
    cd = os.path.dirname(os.path.abspath(__file__)).strip('tests') + 'data'
    sample_data = os.path.join(cd,'Toll_CHC_November_Sample_Data.csv')
    CHC_data = os.path.join(cd,'christchurch_street.csv')
    lat, lon = get_sample(10, 0, '', sample_data, CHC_data, False)
    dm,tm = osrm_get_dist('', '', lat, lon, host='0.0.0.0:5000', save=False)
    dm = np.array(dm)
    tm = np.array(tm)

    delivery_time_windows = {}
    for i in range(1, len(lat)):
        if np.random.rand() > 0.5:
            delivery_time_windows[i] = [0,14400]
        else:
            delivery_time_windows[i] = [14400,28800]

    locs = dm.shape[0]    
    r = ORToolsRouting(locs, 3)
    dim,ind = r.add_dimension(dm, 0, 50000, True, 'distance')
    r.routing.SetArcCostEvaluatorOfAllVehicles(ind)
    dim,ind = r.add_time_windows(tm, delivery_time_windows, 28800, 28800, True, 'time')
    for i in range(1,locs):
        r.add_disjunction(i,50000)
    s = r.solve()
    print(s)

    #s.plot(dm)

def test_orsm_multiday():
    # Setup input files
    cd = os.path.dirname(os.path.abspath(__file__)).strip('tests') + 'data'
    sample_data = os.path.join(cd,'Toll_CHC_November_Sample_Data.csv')
    CHC_data = os.path.join(cd,'christchurch_street.csv')
    
    delivery_lats = np.array([-43.5111688])
    delivery_lons = np.array([172.7319266])

    delivery_time_windows = {}

    for day in range(10):

        # Generate new packages and distance matrix
        lat, lon = get_sample(10, 0, '', sample_data, CHC_data, False)
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
        
        s = r.solve()

        undelivered = np.ones(locs, dtype=bool)
        for route in s.routes:
            for node in route:
                if node != 0:
                    undelivered[node] = False

        delivery_lats = delivery_lats[undelivered]
        delivery_lons = delivery_lons[undelivered]
        print(len(delivery_lats))









if __name__ == "__main__":
    test_orsm_multiday()