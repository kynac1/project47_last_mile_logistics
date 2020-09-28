
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
import osmnx as ox
import networkx as nx

# import plotly.graph_objects as go
import math


# def centroid(n, lat, lon):
#     # coord = [
#     #         ", ".join(str(x) for x in y)
#     #         for y in map(list, [lat, lon].values)
#     #     ]

#     coord = np.array(list(zip(lat, lon))).reshape(len(lat), 2)
#     print(coord)
#     # KMeans algorithm 
#     kmeans_model = KMeans(n_clusters=n).fit(coord)

#     centers = np.array(kmeans_model.cluster_centers_)
#     closest, _ = pairwise_distances_argmin_min(kmeans_model.cluster_centers_, coord)
#     # get the coordinate that is the cloest to its respective centriod
#     closest_coord = coord[closest]
#     plt.plot()
#     plt.title('k means centroids')

#     plt.scatter(coord[:,0],coord[:,1], s = 3, c= kmeans_model.labels_, cmap='rainbow')
#     plt.scatter(centers[:,0], centers[:,1], marker="x", color='black')
#     plt.scatter(closest_coord[:,0], closest_coord[:,1], s = 3, color='black')

#     plt.show()

#     lat, lon = closest_coord.transpose()
#     return list(lat), list(lon)

def centroid_loc_sample(rg, cd, sample_df, sample_sub_dict, CHC_df_grouped, CHC_sub_dict, gp_addr, save):
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
    df_addr = pd.DataFrame(columns=['lat', 'lon', 'address'])
    for sub in sample_sub_dict.keys():
        # fill in address deets
        if sub not in CHC_sub_dict.keys():
            print('no match')
        # get a random number with the size of the suburb
        rd2 = rg.integers(low=0, high=CHC_sub_dict[sub] - 1, size = sample_sub_dict[sub])
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
    df_addr['lat'] = latitude
    df_addr['lon'] = longitude
    df_addr['address'] = address

    # group by street
    if gp_addr == True:
        coord_closest = []
        # set up a dataframe grouped by street
        df_addr[['number','street']] = df_addr.address.str.split(" ", n=1, expand=True)
        addr_gb = df_addr.groupby('street')
        sample_street = list(addr_gb.groups.keys())
        sample_street_size = addr_gb.size().tolist()
        sample_street_dict = dict(zip(sample_street, sample_street_size))

        for street in sample_street_dict.keys():
            street_gp = addr_gb.get_group(street)
            lat_street = street_gp['lat'].values
            lon_street = street_gp['lon'].values
            coord_street = np.array(list(zip(lat_street, lon_street))).reshape(len(lat_street), 2)
            if sample_street_dict[street]!=1:
                # find the centroid of each suburb
                centroid_street = centeroidnp(lat_street, lon_street)
                # find the location in each sample suburb that's closest to its corresponding centroid
                coord_closest.append(closest_node(centroid_street, coord_street))
            else: 
                coord_closest.append(coord_street)
        sample_sub_dict = sample_street_dict
    # sample_sub.pop(0)  # remove empty string
    # # get number of addresses for each suburb
    # sample_sub_size = sample_gb.size().tolist()
    # sample_sub_size.pop(0)  # remove size of empty string

    # a list of centriod locations for all suburbs
    lat_cen, lon_cen = np.vstack(coord_closest).transpose()
    # compute weight 
    weight = list(sample_sub_dict.values())
    sum_w = sum(weight)
    weight = [w/sum_w for w in weight]

    return list(lat_cen), list(lon_cen), weight

def centeroidnp(latitude, longitude):
    length = len(latitude)
    return latitude.sum()/length, longitude.sum()/length
def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    #  dist_2 = np.sum((nodes - node)**2, axis=1)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
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
sample_df, sample_sub_dict, CHC_df, CHC_df_grouped, CHC_sub_dict = read_data(
    sample_data,
    CHC_data,
    lat_min=-43.6147000,
    lat_max=-43.4375000,
    lon_min=172.4768000,
    lon_max=172.7816000,
)

demand = list(sample_sub_dict.values())

# get the centroid coord for each suburb from TOLL sample data
lat, lon, weight = centroid_loc_sample(rg, cd, sample_df, sample_sub_dict, CHC_df_grouped, CHC_sub_dict, gp_addr = False, save=False)
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
    host="0.0.0.0:5000",)

# number of collection points
k = 2
CUSTOMERS = np.arange(len(lat))
FACILITY = np.arange(len(fac_lat))

cap = math.ceil(sample_df.shape[0]/k)
Fac_cap = np.ones(len(fac_lat))* cap

sol_fac_coord, coord, fac_coord = find_opt_collection(k,CUSTOMERS, FACILITY, dist, weight, demand, Fac_cap)

coord = list(map(list, zip(lat, lon)))
fac_coord = list(map(list, zip(fac_lat, fac_lon)))

#  lat = CHC_df["gd2000_ycoord"].array
# lon = CHC_df["gd2000_xcoord"].array

# latitude, longitude = get_sample(
#     5*k, rg, cd, sample_df, sample_sub_dict, CHC_df_grouped, CHC_sub_dict, save=False
# )

############################## PLOT ###################################
# G = nx.DiGraph()
# positions = False
# pos = {}
# if positions:
#             pos = {i: positions[i] for i in range(len(positions))}
# # else:
# #     pos = nx.spring_layout()

# nx.draw(G, pos, with_labels=True)
# ox.plot_graph(G)
# # Downloading the map as a graph object 
# G = ox.graph_from_bbox(north, south, east, west, network_type = 'drive')  
# # Plotting the map graph 
# ox.plot_graph(G)

# lat, lon = centroid(k, latitude, longitude)

# lat, lon = get_sample_per_CHC_suburb(rg, CHC_df_grouped, CHC_sub_dict)


