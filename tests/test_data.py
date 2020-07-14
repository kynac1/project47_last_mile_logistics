from project47.data import *

def test_get_sample_performance():

    cd = os.path.dirname(os.path.abspath(__file__)).strip('tests') + 'data'
    sample_data_csv = os.path.join(cd,'Toll_CHC_November_Sample_Data.csv')
    CHC_data_csv = os.path.join(cd,'christchurch_street.csv')
    sample_df, CHC_df, CHC_sub, CHC_sub_dict = read_data(sample_data_csv, CHC_data_csv)

    lats = []
    lons = []
    for _ in range(100):
        latitude, longitude = get_sample(20, 0, cd, sample_df, CHC_df, CHC_sub, CHC_sub_dict, save=False)
        lats.append(latitude)
        lons.append(longitude)

if __name__ == "__main__":
    test_get_sample_performance()