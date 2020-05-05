# https://maps.googleapis.com/maps/api/geocode/json?address=SOME_ADDRESS&key=YOUR_API_KEY#

import numpy as np
import random

from shapely.geometry import Polygon, Point

import pandas as pd 
from googlemaps import Client as GoogleMaps
from geopy.extra.rate_limiter import RateLimiter

from pandas import DataFrame 

gmaps = GoogleMaps('AIzaSyASm62A_u5U4Kcp4ohOA9lLLXy6PyceT4U')
# df = pd.DataFrame({'Last': ['Smith', 'Nadal', 'Federer'],
#                    'First': ['Steve', 'Joe', 'Roger'],
#                  'Age':[32,34,36]})


df = pd.read_csv("TOLLaddresses.csv")
df['Address'] = df['Street']+ ',' + df['Suburb'] + ',' + df['City'] + ',' + df['Country']

df['long'] = ""
df['lat'] = ""


for x in range(len(df)):
    geocode_result = gmaps.geocode(df['Address'][x])
    df['lat'][x] = geocode_result[0]['geometry']['location'] ['lat']
    df['long'][x] = geocode_result[0]['geometry']['location']['lng']
df.head()
print(df.head())
df.to_csv('TOLLaddress_coords.csv')

'''

# 1 - conveneint function to delay between geocoding calls
geocode = RateLimiter(locator.geocode, min_delay_seconds=1)
# 2- - create location column
df['location'] = df['Address'].apply(geocode)
# 3 - create longitude, laatitude and altitude from location column (returns tuple)
df['point'] = df['location'].apply(lambda loc: tuple(loc.point) if loc else None)
# 4 - split point column into latitude, longitude and altitude columns
df[['latitude', 'longitude', 'altitude']] = pd.DataFrame(df['point'].tolist(), index=df.index)
'''

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