from project47.data import *
from project47.routing import *

def test_ortools_with_osrm():
    cd = os.path.dirname(os.path.abspath(__file__)).strip('tests') + 'data'
    sample_data = os.path.join(cd,'Toll_CHC_November_Sample_Data.csv')
    CHC_data = os.path.join(cd,'christchurch_street.csv')
    get_sample(10, cd, sample_data, CHC_data)
    coord_filename = os.path.join(cd, 'random_subset.csv')
    dm,tm = osrm_get_dist(cd, coord_filename, host='0.0.0.0:5000', save=True)
    dm = np.array(dm)
    locs = dm.shape[0]    
    r = ORToolsRouting(locs, 3)
    dim,ind = r.add_dimension(dm, 0, 50000, True, 'distance')
    r.routing.SetArcCostEvaluatorOfAllVehicles(ind)
    s = r.solve()
    print(s)
    #s.plot(dm)

if __name__ == "__main__":
    test_ortools_with_osrm()