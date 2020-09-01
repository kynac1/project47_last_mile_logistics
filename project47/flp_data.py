
import numpy as np
from project47.routing import *
from project47.customer import Customer
from project47.data import *
from project47.multiday_simulation import *
from numpy.random import Generator, PCG64
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances_argmin_min

def centroid(n, lat, lon):
    # coord = [
    #         ", ".join(str(x) for x in y)
    #         for y in map(list, [lat, lon].values)
    #     ]

    coord = np.array(list(zip(lat, lon))).reshape(len(lat), 2)
    print(coord)
    # KMeans algorithm 
    kmeans_model = KMeans(n_clusters=n).fit(coord)

    centers = np.array(kmeans_model.cluster_centers_)
    closest, _ = pairwise_distances_argmin_min(kmeans_model.cluster_centers_, coord)
    # get the coordinate that is the cloest to its respective centriod
    closest_coord = coord[closest]
    plt.plot()
    plt.title('k means centroids')

    plt.scatter(coord[:,0],coord[:,1], s = 3, c= kmeans_model.labels_, cmap='rainbow')
    plt.scatter(centers[:,0], centers[:,1], marker="x", color='black')
    plt.scatter(closest_coord[:,0], closest_coord[:,1], s = 3, color='black')

    plt.show()

    lat, lon = closest_coord.transpose()
    return list(lat), list(lon)

def sample_centroid(n, rg, cd, sample_df, sample_sub_dict, CHC_df_grouped, CHC_sub_dict, save):
    """
    rg: np.random.Generator
    CHC_df_grouped: CHC data frame - output of 'read_data' function

    This function is to complete the street address for the sample data by sourcing the street address 
    from the Christchurch suburb.
    """
    latitude = []
    longitude = []
    lat_closest = []
    lon_closest = []
    coord_closest = []
    
    for sub in sample_sub_dict.keys():
        if sub not in CHC_sub_dict.keys():
            print('no match')
        # get a random number with the size of the suburb
        rd2 = rg.integers(low=0, high=CHC_sub_dict[sub] - 1, size = sample_sub_dict[sub])
        # randomly pick an address from CHC data based on the suburb
        CHC_row = CHC_df_grouped.get_group(sub).iloc[rd2]  # sample(n=1)
        # fill in address deets
        # row["Receiver Addr2"] = CHC_row["full_address"].values[0]
        # find the location in each sample suburb that's closest to its corresponding centroid
        lat_sub = CHC_row["gd2000_ycoord"].values
        lon_sub = CHC_row["gd2000_xcoord"].values
        coord_sub = np.array(list(zip(lat_sub, lon_sub))).reshape(len(lat_sub), 2)
        centroid_sub = centeroidnp(lat_sub, lon_sub)
        coord_closest.append(closest_node(centroid_sub, coord_sub))

        latitude += list(lat_sub)
        longitude += list(lon_sub)

    return latitude, longitude

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


# number of collection points
k = 10

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


latitude, longitude = sample_centroid(k, rg, cd, sample_df, sample_sub_dict, CHC_df_grouped, CHC_sub_dict, save=False)
# lat = CHC_df["gd2000_ycoord"].array
# lon = CHC_df["gd2000_xcoord"].array

# latitude, longitude = get_sample(
#     5*k, rg, cd, sample_df, sample_sub_dict, CHC_df_grouped, CHC_sub_dict, save=False
# )

lat, lon = centroid(k, latitude, longitude)

# lat, lon = get_sample_per_CHC_suburb(rg, CHC_df_grouped, CHC_sub_dict)
lat_all = fac_lat + lat
lon_all = fac_lon + lon

source = np.arange(len(fac_lat))
# source = np.arange(3)
dist, tm = osrm_get_dist(
    cd,
    coord_filename,
    lat_all,
    lon_all,
    source,
    save=False,
    host="0.0.0.0:5000",)


CUSTOMERS = np.arange(len(lat))
FACILITY = np.arange(len(fac_lat))

Fac_cap = np.ones(len(fac_lat))*20
# Fac_cap = dict(zip(FACILITY, np.ones(len(fac_lat))*20))
# Fac_cost = dict(zip(FACILITY, np.ones(len(fac_lat))*20))

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



