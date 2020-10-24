# https://maps.googleapis.com/maps/api/geocode/json?address=SOME_ADDRESS&key=YOUR_API_KEY#

import numpy as np
import random
import matplotlib.pyplot as plt

# import geopandas as gpd
from shapely.geometry import Polygon, Point
import pandas as pd
import googlemaps
from geopy.extra.rate_limiter import RateLimiter
import requests
import json
import os
import re
from pandas import DataFrame
from numpy.random import Generator, PCG64

import logging

logger = logging.getLogger(__name__)


def read_data(
    sample_data_csv,
    CHC_data_csv,
    lat_min=-1000,
    lat_max=1000,
    lon_min=-1000,
    lon_max=1000,
):
    """
    read in sample data and CHC data in csv once
    return a list of unique suburb names in CHC
    returns a dictionary of number of streets for each suburb
    """
    # read in CHC data
    CHC_df = pd.read_csv(CHC_data_csv, keep_default_na=False)
    # select useful columns
    CHC_df = CHC_df[
        ["suburb_locality", "full_address", "gd2000_ycoord", "gd2000_xcoord"]
    ]
    CHC_df["suburb_locality"] = CHC_df["suburb_locality"].str.upper()

    # exclude the addresses that are out of bounds
    CHC_df = CHC_df[CHC_df["gd2000_ycoord"] >= lat_min]
    CHC_df = CHC_df[CHC_df["gd2000_ycoord"] <= lat_max]
    CHC_df = CHC_df[CHC_df["gd2000_xcoord"] >= lon_min]
    CHC_df = CHC_df[CHC_df["gd2000_xcoord"] <= lon_max]

    # group CHC data by suburbs
    CHC_df_grouped = CHC_df.groupby("suburb_locality")
    # get a list of unique suburb names
    CHC_sub = list(CHC_df_grouped.groups.keys())
    CHC_sub.pop(0)  # remove empty string
    # get number of addresses for each suburb
    CHC_sub_size = CHC_df_grouped.size().tolist()
    CHC_sub_size.pop(0)  # remove size of empty string
    # a dict of sub name and number of streets in each sub
    CHC_sub_dict = dict(zip(CHC_sub, CHC_sub_size))

    # read in Toll's sample data
    sample_df = pd.read_csv(sample_data_csv, keep_default_na=False)
    # filter out data with empty suburbs
    sample_df = sample_df[(sample_df["Receiver Suburb"] != "")]
    # clean suburbs in sample data
    for index, row in sample_df.iterrows():
        sub = re.sub(r"\(.*\)", "", row["Receiver Suburb"]).rstrip()
        row["Receiver Suburb"] = sub
    sample_gb = sample_df.groupby("Receiver Suburb")
    sample_sub = list(sample_gb.groups.keys())
    sample_sub.pop(0)  # remove empty string
    # get number of addresses for each suburb
    sample_sub_size = sample_gb.size().tolist()
    sample_sub_size.pop(0)  # remove size of empty string
    # a dict of sub name and number of streets in each sub
    sample_sub_dict = dict(zip(sample_sub, sample_sub_size))
    # exclude all samples that do not match CHC suburbs
    for sub in sample_sub:
        if sub not in CHC_sub:
            del sample_sub_dict[sub]
    sample_clean = sample_df[sample_df["Receiver Suburb"].isin(
        sample_sub_dict.keys())]

    # ******************************** fall back if grouping does not speed up *************************
    # # extract a list of unique suburbs from CHC_df
    # CHC_sub = CHC_df["suburb_locality"].drop_duplicates().tolist()
    # # remove empty string
    # CHC_sub.remove('')
    # ******************************** fall back if grouping does not speed up *************************

    return sample_clean, sample_sub_dict, CHC_df, CHC_df_grouped, CHC_sub_dict


def get_sample(
    n, rg, cd, sample_df, sample_sub_dict, CHC_df_grouped, CHC_sub_dict, save
):
    """
    n: sample size
    rg: np.random.Generator
    cd: current directory for the use of saving files
    sample_df: sample data frame - output of 'read_data' function
    CHC_df: CHC data frame - output of 'read_data' function
    save: option of saving files

    This function is to get a sample of suburbs from the sample_df and
    then randomly get the street address in the CHC_df according
    to the sample suburbs

    """
    # TOLLdata = pd.read_csv(sample_data, keep_default_na=False)
    # CHCstreet = pd.read_csv(CHC_data, keep_default_na=False)

    # extract random sample of suburbs from sample_df
    rd = rg.integers(low=0, high=len(sample_df) - 1,
                     size=n)  # a list of random numbers
    random_subset = sample_df.iloc[rd]  # sample(n)

    latitude = []
    longitude = []
    # x = 0
    # coordinates = []

    for index, row in random_subset.iterrows():
        # clean suburbs - get rid of () and things within
        sub = row["Receiver Suburb"]
        # # if the suburb does not exsit in CHC data, get a new sample point that exists
        # while len(CHC_df[CHC_df["suburb_locality"].str.upper() == sub]) == 0:
        # while row["Receiver Suburb"] not in CHC_sub:
        #     x += 1
        # # get a new random row
        # rd1 = rg.integers(low=0, high=len(sample_df) - 1, size=1)
        # row = sample_df.iloc[rd1]
        # row = sample_df.sample(n=1)
        # sub = re.sub(r"\(.*\)", "", row["Receiver Suburb"].values[0]).rstrip()
        # row["Receiver Suburb"] = sub
        # print(x)
        # get a random number with the size of the suburb
        rd2 = rg.integers(low=0, high=CHC_sub_dict[sub] - 1, size=1)
        # randomly pick an address from CHC data based on the suburb
        CHC_row = CHC_df_grouped.get_group(sub).iloc[rd2]  # sample(n=1)
        # fill in address deets
        row["Receiver Addr2"] = CHC_row["full_address"].values[0]
        latitude.append(CHC_row["gd2000_ycoord"].values[0])
        longitude.append(CHC_row["gd2000_xcoord"].values[0])

        # ******************************** fall back if grouping does not speed up *************************
        # filtre on the same suburb in CHC street data
        # rd2 = rg.integers(low=0, high=len(CHC_df[CHC_df["suburb_locality"] == sub])-1, size=1)
        # CHC_row = CHC_df[CHC_df["suburb_locality"] == sub].iloc[rd2] #sample(n=1)
        # row["Receiver Addr2"] = CHC_row["full_address"].values[0]
        # latitude.append(CHC_row["gd2000_ycoord"].values[0])
        # longitude.append(CHC_row["gd2000_xcoord"].values[0])
        # ******************************** fall back if grouping does not speed up *************************

        # coordinates.append(str(CHC_row["gd2000_ycoord"].values[0])+ ', ' + str(CHC_row["gd2000_xcoord"].values[0]))
    # save to a file if required
    if save:
        df = pd.DataFrame(random_subset)
        df["latitude"] = latitude
        df["longitude"] = longitude
        # df["coordinates"] = coordinates
        df.to_csv(os.path.join(cd, "random_subset.csv"))
    return latitude, longitude


def get_coordinates(API_key, cd, address_filename, coord_filename, save=False):
    gmaps = googlemaps.Client(key=API_key)

    # # Requires cities name
    # my_dist = gmaps.distance_matrix('Delhi','Mumbai')['rows'][0]['elements'][0]
    df = pd.read_csv(address_filename, keep_default_na=False)
    df["Address"] = (
        df["Street"] + "," + df["Suburb"] +
        "," + df["City"] + "," + df["Country"]
    )

    latitude = []
    longitude = []
    # df['latitude'] = ""
    # df['longitude'] = ""
    # get longitude and latitude using the address
    for x in range(len(df)):
        geocode_result = gmaps.geocode(df["Address"][x])
        latitude.append(geocode_result[0]["geometry"]["location"]["lat"])
        longitude.append(geocode_result[0]["geometry"]["location"]["lng"])
        # df['latitude'][x] = geocode_result[0]['geometry']['location'] ['lat']
        # df['longitude'][x] = geocode_result[0]['geometry']['location']['lng']
    # combine latitude and longitude into coordinates
    # df['coordinates'] = [', '.join(str(x) for x in y) for y in map(tuple, df[['latitude', 'longitude']].values)]
    # save to a file if required
    if save:
        df["latitude"] = latitude
        df["longitude"] = longitude
        # df["coordinates"] = coordinates
        df.to_csv(os.path.join(cd, coord_filename))
    return latitude, longitude
    # return


def get_dist(API_key, cd, coord_filename, latitude, longitude, save=False):
    gmaps = googlemaps.Client(key=API_key)
    # get destination according to input data types
    if len(latitude) != 0 and len(longitude) != 0:
        # combine latitude and longitude into coordinates
        destinations = list(zip(latitude, longitude))
    elif coord_filename != "" and coord_filename != None:
        # read in coodinates
        data = pd.read_csv(coord_filename, keep_default_na=False)
        # combine latitude and longitude into coordinates
        data["coordinates"] = [
            ", ".join(str(x) for x in y)
            for y in map(tuple, data[["latitude", "longitude"]].values)
        ]
        destinations = data.coordinates
    else:
        logger.error("No Input Data")
        # print("Warning: No input data")
        return [], []

    # get distance (km) and durantion (hr) matrix
    def result(p1, p2): return gmaps.distance_matrix(p1, p2, mode="driving")["rows"][0][
        "elements"
    ][0]
    dm = np.asarray(
        [
            [result(p1, p2)["distance"]["value"] for p2 in destinations]
            for p1 in destinations
        ]
    )
    tm = np.asarray(
        [
            [result(p1, p2)["duration"]["value"] for p2 in destinations]
            for p1 in destinations
        ]
    )

    if save:
        df = pd.DataFrame(dm)
        df.to_csv(os.path.join(cd, "dm.csv"),
                  float_format="%.3f", na_rep="NAN!")
        df = pd.DataFrame(tm)
        df.to_csv(os.path.join(cd, "tm.csv"),
                  float_format="%.3f", na_rep="NAN!")
    return dm, tm


def osrm_get_dist(
    cd,
    coord_filename,
    latitude,
    longitude,
    source=[],
    save=False,
    host="router.project-orsm.org",
):
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
        logger.error("No Input Data")
        # print("Warning: No input data")
        return [], []

    # set up request
    dest_string = ""
    for i in destinations:
        dest_string = dest_string + i + ";"
    dest_string = dest_string.rstrip(";")
    url = "http://" + host + "/table/v1/driving/" + dest_string
    if local:
        # + destinations[0]+";" +destinations[1]+";" +destinations[2] #+ '?annotations=distance'
        url += "?annotations=distance,duration"
    if len(source) != 0:
        url += "&sources="
        for i in source[:-1]:
            url += f"{i};"
        url += f"{source[-1]}"
        url += "&destinations=" + ";".join(
            f"{i}" for i in range(len(latitude)) if i not in source
        )
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
        logger.critical("OSRM request failed")
        logger.critical("%s" % result)
        return None, None


def main():
    # API_key = "AIzaSyASm62A_u5U4Kcp4ohOA9lLLXy6PyceT4U"
    cd = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data"
    )  # direct to data folder
    sample_data_csv = os.path.join(cd, "Toll_CHC_November_Sample_Data.csv")
    CHC_data_csv = os.path.join(cd, "christchurch_street.csv")
    sample_df, sample_sub_dict, CHC_df, CHC_df_grouped, CHC_sub_dict = read_data(
        sample_data_csv, CHC_data_csv
    )
    seed = 123456789
    rg = Generator(PCG64(seed))

    latitude, longitude = get_sample(
        100,
        rg,
        cd,
        sample_df,
        sample_sub_dict,
        CHC_df_grouped,
        CHC_sub_dict,
        save=False,
    )
    # get_sample(n, rg, cd, sample_df, CHC_df, CHC_sub, CHC_sub_dict, save)
    # get a random sample of locations in Christchurch
    # get_sample(10, cd, sample_data, CHC_data)
    # latitude, longitude = get_sample(5, 0, cd, sample_data, CHC_data, save=False)
    # latitude, longitude = '', ''

    coord_filename = os.path.join(cd, "random_subset.csv")
    # get_coordinates(API_key, cd, address_filename, coord_filename)
    # coord_filename = None
    # dm, tm = get_dist(API_key, cd, coord_filename, latitude, longitude, save=False)
    dm, tm = osrm_get_dist(
        cd, coord_filename, latitude, longitude, host="localhost:5000", save=True
    )
    # print(osrm_get_dist(cd, coord_filename, host='localhost:5000', save=True))
    # print(dm)
    # print(tm)


if __name__ == "__main__":
    main()

# # coordinate reference system to earth
# crs={'init':'epsg:4326'}
# # define the geometry
# geometry=[Point(xy) for xy in zip(data["long"], data["lat"])]

# geodata=gpd.GeoDataFrame(data,crs=crs, geometry=geometry)
# geodata.plot()

# Calculate the the length of the shortest path between 2 points on the earth.
# It calculates geodesic distance from latitude-longitude data.
"""
# Importing the geodesic module from the library 
from geopy.distance import geodesic 
# Loading the lat-long data for Kolkata & Delhi 
d1 = (-43.5111688,172.7319266) 
d2 = (-43.5499101, 172.63913) 
# Print the distance calculated in km 
print(geodesic(d1, d2).km) 
"""


# # 1 - conveneint function to delay between geocoding calls
# geocode = RateLimiter(locator.geocode, min_delay_seconds=1)
# # 2- - create location column
# df['location'] = df['Address'].apply(geocode)
# # 3 - create longitude, laatitude and altitude from location column (returns tuple)
# df['point'] = df['location'].apply(lambda loc: tuple(loc.point) if loc else None)
# # 4 - split point column into latitude, longitude and altitude columns
# df[['latitude', 'longitude', 'altitude']] = pd.DataFrame(df['point'].tolist(), index=df.index)


"""
# generate random points within a ploygon

# poly = Polygon([(23.789642, 90.354714), (23.789603, 90.403000), (23.767688, 90.403597),(23.766510, 90.355448)])
poly = Polygon([(23.789642, 90.354714), (23.789603, 90.403000), (23.767688, 90.403597),(23.766510, 90.355448)])


def random_points_within(poly, num_points):
    min_x, min_y, max_x, max_y = poly.bounds

    points = []

    while len(points) < num_points:
        random_point = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
        if (random_point.within(poly)):
            points.append(random_point)

    return points


points = random_points_within(poly,1000)

for p in points:
    print(p.x,",",p.y)
"""
