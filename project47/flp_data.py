
import numpy as np
from project47.routing import *
from project47.customer import Customer
from project47.data import *
from project47.multiday_simulation import *
from numpy.random import Generator, PCG64
import os
import matplotlib.pyplot as plt

# def sample_generator(rg: np.random.Generator):
#     cd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
#     sample_data = os.path.join(cd, "Toll_CHC_November_Sample_Data.csv")
#     CHC_data = os.path.join(cd, "christchurch_street.csv")
#     sample_df, CHC_df, CHC_sub, CHC_sub_dict = read_data(
#         sample_data,
#         CHC_data,
#         lat_min=-43.6147000,
#         lat_max=-43.4375000,
#         lon_min=172.4768000,
#         lon_max=172.7816000,
#     )

#     lat, lon = get_sample(
#         100, rg, cd, sample_df, CHC_df, CHC_sub, CHC_sub_dict, save=False
#     )

def osrm_get_fac_dist(cd, coord_filename, latitude, longitude, source_num, save=False, host="router.project-orsm.org"):
    local = (
        host != "router.project-orsm.org"
    )  # We assume it's local, and can get distances
    # get destination according to input data types
    if len(latitude) != 0 and len(longitude) != 0:
        # combine latitude and longitude into coordinates
        destinations = [
            ",".join(str(x) for x in y) for y in map(list, zip(longitude, latitude))
        ]
    elif coord_filename != "" and coord_filename != None:
        # read in coodinates
        data = pd.read_csv(coord_filename, keep_default_na=False)
        # combine latitude and longitude into coordinates
        data["coordinates"] = [
            ",".join(str(x) for x in y)
            for y in map(tuple, data[["longitude", "latitude"]].values)
        ]
        destinations = data.coordinates
    else:
        print("Warning: No input data")
        return [], []
    
    # set up request
    dest_string = ""
    if source_num == 0:
        for i in destinations:
            dest_string = dest_string + i + ";"
    else:
        index_string =""
        index = 0
        for i in destinations:
            dest_string = dest_string + i + ";"
            index_string = index_string + str(index) + ";"
            index += 1
        source_ind, dest_ind = index_string.split(sep=";"+str(source_num), maxsplit=1)[:]
        dest_ind = str(source_num)+dest_ind
        dest_ind = dest_ind.rstrip(";")
    
    dest_string = dest_string.rstrip(";")
    url = "http://" + host + "/table/v1/driving/" + dest_string
    if source_num != 0:
        url = url + "?sources=" + source_ind#+"&destinations="+dest_ind
    if local and source_num == 0:
        url += "?annotations=distance,duration"  # + destinations[0]+";" +destinations[1]+";" +destinations[2] #+ '?annotations=distance'
    response = requests.get(url)
    result = response.json()
    if result["code"] == "Ok":
        tm = result["durations"]
        # round times into int
        tm[:] = [[round(y) for y in x] for x in tm]
        if local:
            dm = result["distances"]
            # round distances into int
            dm[:] = [[round(y) for y in x] for x in dm]
        if save:
            if local:
                df = pd.DataFrame(dm)
                df.to_csv(
                    os.path.join(cd, "dm_orsm.csv"), float_format="%.3f", na_rep="NAN!"
                )
            df = pd.DataFrame(tm)
            df.to_csv(
                os.path.join(cd, "tm_osrm.csv"), float_format="%.3f", na_rep="NAN!"
            )
        if local:
            return dm, tm
        else:
            return None, tm
    else:
        return None, None

# API_key = "AIzaSyASm62A_u5U4Kcp4ohOA9lLLXy6PyceT4U"
cd = (os.path.dirname(os.path.abspath(__file__)).strip("project47") + "data")  # direct to data folder
# address_filename = os.path.join(cd, "fac_addresses.csv")
coord_filename = os.path.join(cd, "fac_coord.csv")
# fac_lat, fac_lon = get_coordinates(API_key, cd, address_filename, coord_filename, save=True)
df = pd.read_csv(coord_filename, keep_default_na=False)
fac_lat = df["latitude"].tolist()
fac_lon = df["longitude"].tolist()

seed = 123456789
rg = Generator(PCG64(seed))
sample_data = os.path.join(cd, "Toll_CHC_November_Sample_Data.csv")
CHC_data = os.path.join(cd, "christchurch_street.csv")
sample_df, CHC_df, CHC_sub, CHC_sub_dict = read_data(
    sample_data,
    CHC_data,
    lat_min=-43.6147000,
    lat_max=-43.4375000,
    lon_min=172.4768000,
    lon_max=172.7816000,
)
lat, lon = get_sample(
        100, rg, cd, sample_df, CHC_df, CHC_sub, CHC_sub_dict, save=False
    )

lat_all = fac_lat + lat
lon_all = fac_lon + lon
source_num = len(fac_lat)
dm, tm = osrm_get_fac_dist(
    cd, coord_filename, lat_all, lon_all, source_num, save=False, host="router.project-orsm.org")
CUSTOMERS = np.arange(len(lat))
FACILITY = np.arange(len(fac_lat))
Fac_cap = dict(zip(FACILITY, np.ones(len(fac_lat)*20)))
Fac_cost = dict(zip(FACILITY, np.ones(len(fac_lat)*20)))
# CUSTOMERS = [1,2,3,4,5]
# FACILITY = ['f1','f2','f3']
# Fac_cost = {'f1': 5,
#             'f2': 10,
#             'f3': 10}

# Fac_cap = {'f1': 50,
#             'f2': 50,
#             'f3': 50}

# dist = {'f1': {1: 4, 2: 5, 3: 6, 4: 8, 5: 10},
#         'f2': {1: 6, 2: 4, 3: 3, 4: 5, 5: 8},
#         'f3': {1: 9, 2: 7, 3: 4, 4: 3, 5: 4}}


FACILITY = []
Fac_address
Fac_loc = get_coordinates(API_key, cd, address_filename, coord_filename, save=False)
Fac_cost = {j: 10000 for j in FACILITY}
Fac_cap = {j: 50 for j in FACILITY}
lat = [customer.lat for customer in customers],
lon = [customer.lon for customer in customers],
dist = osrm_get_dist(
        "",
        "",
        [customer.lat for customer in customers],
        [customer.lon for customer in customers],
        host="0.0.0.0:5000",
        save=False,
    )