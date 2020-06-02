# https://maps.googleapis.com/maps/api/geocode/json?address=SOME_ADDRESS&key=YOUR_API_KEY#

import numpy as np
import random
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon, Point
import pandas as pd 
import googlemaps 
from geopy.extra.rate_limiter import RateLimiter
import requests
import json
import os
import re
from pandas import DataFrame 

def get_sample(n, seed, cd, sample_data, CHC_data, save):
    '''
    n: sample size
    '''
    np.random.seed(seed)
    
    TOLLdata = pd.read_csv(sample_data, keep_default_na=False)
    CHCstreet = pd.read_csv(CHC_data, keep_default_na=False)
    rd = np.random.randint(low=0, high=len(TOLLdata)-1, size=n)
    random_subset = TOLLdata[(TOLLdata["Receiver Suburb"]!= "")].iloc[rd]#sample(n)

    latitude = []
    longitude = []
    # coordinates = []
    CHCstreet["full_address"] = CHCstreet["full_address"].astype(str) 
    # get rid of () and things within
    for index, row in random_subset.iterrows():
        # if missing street address
        # if row["Receiver Addr2"] == '':
        # clean suburbs
        sub = re.sub(r"\(.*\)", "", row["Receiver Suburb"]).rstrip()
        row["Receiver Suburb"] = sub
        # if the suburb does not exsit in CHC data, get a new sample point that exists
        while len(CHCstreet[CHCstreet["suburb_locality"].str.upper() == sub]) == 0:
            row = TOLLdata[(TOLLdata["Receiver Suburb"]!= "")].sample(n=1)
            sub = re.sub(r"\(.*\)", "", row["Receiver Suburb"].values[0]).rstrip()
            row["Receiver Suburb"] = sub
        # filtre on the same suburb in CHC street data
        rd1 = np.random.randint(low=0, high=len(CHCstreet[CHCstreet["suburb_locality"].str.upper() == sub])-1, size=1)
        CHC_row = CHCstreet[CHCstreet["suburb_locality"].str.upper() == sub].iloc[rd1]#sample(n=1) 
        row["Receiver Addr2"] = CHC_row["full_address"].values[0]
        latitude.append(CHC_row["gd2000_ycoord"].values[0])
        longitude.append(CHC_row["gd2000_xcoord"].values[0])
        # coordinates.append(str(CHC_row["gd2000_ycoord"].values[0])+ ', ' + str(CHC_row["gd2000_xcoord"].values[0]))
    # save to a file if required
    if save: 
        df = pd.DataFrame(random_subset)
        df["latitude"] = latitude
        df["longitude"] = longitude
        # df["coordinates"] = coordinates
        df.to_csv(os.path.join(cd,'random_subset.csv'))
    return latitude, longitude


def get_coordinates(API_key, cd, address_filename, coord_filename, save=False):
    gmaps = googlemaps.Client(key=API_key)

    # # Requires cities name 
    # my_dist = gmaps.distance_matrix('Delhi','Mumbai')['rows'][0]['elements'][0] 
    df = pd.read_csv(address_filename, keep_default_na=False)
    df['Address'] = df['Street']+ ',' + df['Suburb'] + ',' + df['City'] + ',' + df['Country']

    latitude = []
    longitude = []
    # df['latitude'] = ""
    # df['longitude'] = ""
    # get longitude and latitude using the address
    for x in range(len(df)):
        geocode_result = gmaps.geocode(df['Address'][x])
        latitude.append(geocode_result[0]['geometry']['location'] ['lat'])
        longitude.append(geocode_result[0]['geometry']['location']['lng'])
        # df['latitude'][x] = geocode_result[0]['geometry']['location'] ['lat']
        # df['longitude'][x] = geocode_result[0]['geometry']['location']['lng']
    # combine latitude and longitude into coordinates
    # df['coordinates'] = [', '.join(str(x) for x in y) for y in map(tuple, df[['latitude', 'longitude']].values)]
    # save to a file if required
    if save: 
        df["latitude"] = latitude
        df["longitude"] = longitude
        # df["coordinates"] = coordinates
        df.to_csv(os.path.join(cd,coord_filename))
    return latitude, longitude
    # return 

def get_dist(API_key, cd, coord_filename, latitude, longitude, save=False):
    gmaps = googlemaps.Client(key=API_key)
    # get destination according to input data types
    if len(latitude)!= 0 and len(longitude)!= 0:
        # combine latitude and longitude into coordinates
        destinations = list(zip(latitude, longitude))
    elif coord_filename != '' and coord_filename != None:
        # read in coodinates
        data = pd.read_csv(coord_filename, keep_default_na=False) 
        # combine latitude and longitude into coordinates
        data['coordinates'] = [', '.join(str(x) for x in y) for y in map(tuple, data[['latitude', 'longitude']].values)]
        destinations = data.coordinates
    else:
        print('Warning: No input data' )
        return [],[]

    # get distance (km) and durantion (hr) matrix
    result = lambda p1, p2: gmaps.distance_matrix(p1, p2, mode='driving')["rows"][0]["elements"][0]
    dm = np.asarray([[result(p1, p2)["distance"]["value"]/1000   for p2 in destinations] for p1 in destinations])
    tm = np.asarray([[result(p1, p2)["duration"]["value"]/3600   for p2 in destinations] for p1 in destinations])
    
    if save: 
        df = pd.DataFrame(dm)
        df.to_csv(os.path.join(cd,'dm.csv'), float_format='%.3f', na_rep="NAN!")
        df = pd.DataFrame(tm)
        df.to_csv(os.path.join(cd,'tm.csv'), float_format='%.3f', na_rep="NAN!")
    return dm, tm

def osrm_get_dist(cd, coord_filename, latitude, longitude, save=False, host='router.project-orsm.org'):
    local = host != 'router.project-orsm.org' # We assume it's local, and can get distances
    # get destination according to input data types
    if len(latitude)!= 0 and len(longitude)!= 0:
        # combine latitude and longitude into coordinates
        destinations = [','.join(str(x) for x in y) for y in map(list, zip(longitude, latitude))]
    elif coord_filename != '' and coord_filename != None:
        # read in coodinates
        data = pd.read_csv(coord_filename, keep_default_na=False) 
        # combine latitude and longitude into coordinates
        data['coordinates'] = [','.join(str(x) for x in y) for y in map(tuple, data[['longitude', 'latitude']].values)]
        destinations = data.coordinates
    else:
        print('Warning: No input data' )
        return [],[]

    # set up request
    dest_string = ''
    for i in destinations:
        dest_string = dest_string + i + ';'
    dest_string = dest_string.rstrip(';') 
    url =  'http://' + host + '/table/v1/driving/' + dest_string 
    if local: url += '?annotations=distance,duration' #+ destinations[0]+";" +destinations[1]+";" +destinations[2] #+ '?annotations=distance'
    response = requests.get(url) 
    result = response.json()
    if result['code'] == 'Ok':
        tm = result['durations']
        if local: dm = result['distances']
        # convert to hrs
        tm[:] = [[y / 3600 for y in x] for x in tm]
        if save:
            if local:
                df = pd.DataFrame(dm)
                df.to_csv(os.path.join(cd,'dm_orsm.csv'), float_format='%.3f', na_rep="NAN!")
            df = pd.DataFrame(tm)
            df.to_csv(os.path.join(cd,'tm_osrm.csv'), float_format='%.3f', na_rep="NAN!")
        if local:
            return dm, tm
        else:
            return None, tm
    else:
        return None, None

def main():
    API_key = 'AIzaSyASm62A_u5U4Kcp4ohOA9lLLXy6PyceT4U'
    cd = os.path.dirname(os.path.abspath(__file__)).strip('project47') + 'data' # direct to data folder
    sample_data = os.path.join(cd,'Toll_CHC_November_Sample_Data.csv')
    CHC_data = os.path.join(cd,'christchurch_street.csv')
    # get a random sample of locations in Christchurch
    # get_sample(10, cd, sample_data, CHC_data)
    latitude, longitude = get_sample(5, 1, cd, sample_data, CHC_data, save=False)
    # latitude, longitude = '', ''

    coord_filename = os.path.join(cd, 'random_subset.csv')
    # get_coordinates(API_key, cd, address_filename, coord_filename)
    # coord_filename = None
    # dm, tm = get_dist(API_key, cd, coord_filename, latitude, longitude, save=False)
    dm, tm = osrm_get_dist(cd, coord_filename, latitude, longitude, host='0.0.0.0:5000', save=True)
    # print(osrm_get_dist(cd, coord_filename, host='0.0.0.0:5000', save=True))
    print(dm)
    print(tm)


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
'''
# Importing the geodesic module from the library 
from geopy.distance import geodesic 
# Loading the lat-long data for Kolkata & Delhi 
d1 = (-43.5111688,172.7319266) 
d2 = (-43.5499101, 172.63913) 
# Print the distance calculated in km 
print(geodesic(d1, d2).km) 
'''


# # 1 - conveneint function to delay between geocoding calls
# geocode = RateLimiter(locator.geocode, min_delay_seconds=1)
# # 2- - create location column
# df['location'] = df['Address'].apply(geocode)
# # 3 - create longitude, laatitude and altitude from location column (returns tuple)
# df['point'] = df['location'].apply(lambda loc: tuple(loc.point) if loc else None)
# # 4 - split point column into latitude, longitude and altitude columns
# df[['latitude', 'longitude', 'altitude']] = pd.DataFrame(df['point'].tolist(), index=df.index)


'''
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
'''