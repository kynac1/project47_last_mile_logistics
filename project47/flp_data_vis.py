import numpy as np
from project47.routing import *
from project47.customer import Customer
from project47.data import *
from project47.flp_func import *

# from project47.multiday_simulation import *
from numpy.random import Generator, PCG64
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances_argmin_min
import networkx as nx
import math

"""
This script is to produce solutions for data visualisation of the collection optimisation problem 
so the data here can be flowed to flp_vis.ipynb to plot graphs.
"""


def centroid_loc_sample(
    rg, cd, sample_df, sample_sub_dict, CHC_df_grouped, CHC_sub_dict, gp_addr, save
):
    """
    rg: np.random.Generator
    CHC_df_grouped: CHC data frame - output of 'read_data' function

    This function is to complete the street address for the sample data by sourcing the street address
    from the Christchurch suburb.
    """
    latitude = []
    longitude = []
    address = []
    coord_closest = []
    df_addr = pd.DataFrame(columns=["lat", "lon", "address"])
    for sub in sample_sub_dict.keys():
        # fill in address deets
        if sub not in CHC_sub_dict.keys():
            print("no match")
        # get a random number with the size of the suburb
        rd2 = rg.integers(low=0, high=CHC_sub_dict[sub] - 1, size=sample_sub_dict[sub])
        # randomly pick an address from CHC data based on the suburb
        CHC_row = CHC_df_grouped.get_group(sub).iloc[rd2]
        lat_sub = CHC_row["gd2000_ycoord"].values
        lon_sub = CHC_row["gd2000_xcoord"].values
        address_sub = CHC_row["full_address"].values
        coord_sub = np.array(list(zip(lat_sub, lon_sub))).reshape(len(lat_sub), 2)

        # find the centroid of each suburb
        centroid_sub = centeroidnp(lat_sub, lon_sub)
        # find the location in each sample suburb that's closest to its corresponding centroid
        coord_closest.append(closest_node(centroid_sub, coord_sub))

        # all sampled locations
        latitude += list(lat_sub)
        longitude += list(lon_sub)
        address += list(address_sub)
    df_addr["lat"] = latitude
    df_addr["lon"] = longitude
    df_addr["address"] = address

    # group by street
    if gp_addr == True:
        coord_closest = []
        # set up a dataframe grouped by street
        df_addr[["number", "street"]] = df_addr.address.str.split(" ", n=1, expand=True)
        addr_gb = df_addr.groupby("street")
        sample_street = list(addr_gb.groups.keys())
        sample_street_size = addr_gb.size().tolist()
        sample_street_dict = dict(zip(sample_street, sample_street_size))

        for street in sample_street_dict.keys():
            street_gp = addr_gb.get_group(street)
            lat_street = street_gp["lat"].values
            lon_street = street_gp["lon"].values
            coord_street = np.array(list(zip(lat_street, lon_street))).reshape(
                len(lat_street), 2
            )
            if sample_street_dict[street] != 1:
                # find the centroid of each suburb
                centroid_street = centeroidnp(lat_street, lon_street)
                # find the location in each sample suburb that's closest to its corresponding centroid
                coord_closest.append(closest_node(centroid_street, coord_street))
            else:
                coord_closest.append(coord_street)
        sample_sub_dict = sample_street_dict

    # a list of centriod locations for all suburbs
    lat_cen, lon_cen = np.vstack(coord_closest).transpose()
    # compute weight
    weight = list(sample_sub_dict.values())
    sum_w = sum(weight)
    weight = [w / sum_w for w in weight]

    return list(lat_cen), list(lon_cen), weight


def centeroidnp(latitude, longitude):
    length = len(latitude)
    return latitude.sum() / length, longitude.sum() / length


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    #  dist_2 = np.sum((nodes - node)**2, axis=1)
    deltas = nodes - node
    dist_2 = np.einsum("ij,ij->i", deltas, deltas)
    return nodes[np.argmin(dist_2)]


def get_sample_per_CHC_suburb(rg, CHC_df_grouped, CHC_sub_dict):
    """
    rg: np.random.Generator
    CHC_df_grouped: CHC data frame - output of 'read_data' function

    This function is to get an address from each of the Christchurch suburb.
    """
    latitude = []
    longitude = []

    for sub in CHC_sub_dict.keys():
        # get a random number with the size of the suburb
        rd2 = rg.integers(low=0, high=CHC_sub_dict[sub] - 1, size=1)
        # randomly pick an address from CHC data based on the suburb
        CHC_row = CHC_df_grouped.get_group(sub).iloc[rd2]  # sample(n=1)
        # fill in address deets
        # row["Receiver Addr2"] = CHC_row["full_address"].values[0]
        latitude.append(CHC_row["gd2000_ycoord"].values[0])
        longitude.append(CHC_row["gd2000_xcoord"].values[0])
    return latitude, longitude


# API_key = "AIzaSyASm62A_u5U4Kcp4ohOA9lLLXy6PyceT4U"
cd = os.path.dirname(os.path.abspath(__file__)) + "/../data"
# "/Users/karen.w/Desktop/project47_last_mile_logistics/datacoord_filename= "/Users/karen.w/Desktop/project47_last_mile_logistics/data/fac_coord.csv" # direct to data folder
coord_filename = os.path.join(cd, "fac_coord.csv")
# direct to collection coordinates file
# get lat and lon for collection points
df = pd.read_csv(coord_filename, keep_default_na=False)
fac_lat = df["latitude"].tolist()
fac_lon = df["longitude"].tolist()

# can prolly be commented out and use the read in files from multiday?
seed = 123456789
rg = Generator(PCG64(seed))
sample_data = os.path.join(cd, "Toll_CHC_November_Sample_Data.csv")
CHC_data = os.path.join(cd, "christchurch_street.csv")
sample_df, sample_sub_dict, CHC_df, CHC_df_grouped, CHC_sub_dict = read_data(
    sample_data,
    CHC_data,
    lat_min=-43.6147000,
    lat_max=-43.4375000,
    lon_min=172.4768000,
    lon_max=172.7816000,
)

# demand of package storage for each suburb
demand = list(sample_sub_dict.values())

# get the centroid coord for each suburb from TOLL sample data
lat, lon, weight = centroid_loc_sample(
    rg,
    cd,
    sample_df,
    sample_sub_dict,
    CHC_df_grouped,
    CHC_sub_dict,
    gp_addr=False,
    save=False,
)
# combine collection and customer locations
lat_all = fac_lat + lat
lon_all = fac_lon + lon
# get the total number of potential collections points
source = np.arange(len(fac_lat))
# contruct the distance matrix from collection point to centroid of each suburb
dist, tm = osrm_get_dist(
    cd,
    coord_filename,
    lat_all,
    lon_all,
    source,
    save=False,
    host="localhost:5000",
)

# number of collection points
k = 2
CUSTOMERS = np.arange(len(lat))
FACILITY = np.arange(len(fac_lat))
cap = 30
# cap = math.ceil(sample_df.shape[0] / k)
Fac_cap = np.ones(len(fac_lat)) * sum(demand)

# solve the location optimisation problem
sol_fac_lat, sol_fac_lon = find_opt_collection(
    k, CUSTOMERS, FACILITY, fac_lat, fac_lon, dist, weight, demand, Fac_cap
)
coord = list(map(list, zip(lat, lon)))
fac_coord = list(map(list, zip(fac_lat, fac_lon)))
