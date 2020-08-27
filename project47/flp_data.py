
import numpy as np
from project47.routing import *
from project47.customer import Customer
from project47.data import *
from project47.multiday_simulation import *
from numpy.random import Generator, PCG64
import os
import matplotlib.pyplot as plt

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
        10, rg, cd, sample_df, CHC_df, CHC_sub, CHC_sub_dict, save=False
    )

lat_all = fac_lat + lat
lon_all = fac_lon + lon

source = np.arange(len(fac_lat))
# source = np.arange(3)
dm, tm = osrm_get_dist(
    cd,
    coord_filename,
    lat_all,
    lon_all,
    source,
    save=False,
    host="0.0.0.0:5000",)


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