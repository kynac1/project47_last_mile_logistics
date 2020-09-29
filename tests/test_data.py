from project47.data import *


def test_get_sample_performance():

    cd = os.path.dirname(os.path.abspath(__file__)).strip("tests") + "data"
    sample_data_csv = os.path.join(cd, "Toll_CHC_November_Sample_Data.csv")
    CHC_data_csv = os.path.join(cd, "christchurch_street.csv")
    sample_df, CHC_df, _, CHC_sub, CHC_sub_dict = read_data(
        sample_data_csv, CHC_data_csv
    )

    lats = []
    lons = []
    for _ in range(10):
        latitude, longitude = get_sample(
            20,
            np.random.Generator(np.random.PCG64()),
            cd,
            sample_df,
            CHC_df,
            CHC_sub,
            CHC_sub_dict,
            save=False,
        )
        lats.append(latitude)
        lons.append(longitude)


def test_osrm():
    """This will only work with the osrm server running, and with the Christchurch map"""

    latitude, longitude = [-43.2, -43.3], [172.5, 172.6]
    res = osrm_get_dist("", "", latitude, longitude, host="0.0.0.0:5000", save=False)
    assert np.allclose(np.array(res[0]), np.array([[0, 6014], [6014, 0]]))
    assert np.allclose(np.array(res[1]), np.array([[0, 292], [290, 0]]))

    latitude, longitude = (
        [-43.2 - 0.1 * i for i in range(10)],
        [172.5 + 0.1 * 2 for i in range(10)],
    )
    # We set the second location (at position 1, zero based index) as the source
    # So we only
    res = osrm_get_dist(
        "", "", latitude, longitude, source=[1], host="0.0.0.0:5000", save=False
    )
    assert len(res[0]) == 1
    assert len(res[1]) == 1
    assert len(res[0][0]) == 9
    assert len(res[1][0]) == 9


if __name__ == "__main__":
    test_osrm()
