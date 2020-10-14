from project47.routing import *
from project47.customer import Customer
from project47.data import get_sample, read_data

# from project47.multiday_simulation import *
# from project47.flp_data import *
# from project47.flp_func import *
from numpy.random import Generator, PCG64
import os
import webbrowser
import matplotlib.pyplot as plt
import folium
import pandas as pd

# import osmnx as ox
import networkx as nx
import numpy as np
from IPython.display import HTML, display

# fig, axs = plt.subplots()
# # plt.scatter(lon,lat, s = 5, c=weight)
# plt.gray()
# plt.scatter( fac_lon, fac_lat, s = 20, c="blue", marker="^", alpha=0.5)
# # plt.scatter( sol_fac_lon, sol_fac_lat, s = 50 , c="red", marker="*")
# # plt.title('Scatter plot pythonspot.com')
# plt.xlabel('lon')
# plt.ylabel('lat')
# plt.show()

# place = “Christchurch, New Zealand”
# graph = ox.graph_from_place(place, network_type=’drive’)
# nodes, streets = ox.graph_to_gdfs(graph)
# streets.head()

# style = {‘color’: ‘#F7DC6F’, ‘weight’:’1'}
cd = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."
)  # direct to data folder

m = folium.Map(location=[-40.9006, 174.8860], zoom_start=15)

m.render()
display(m)


m = folium.Map()
m.save("nz_map.html")
webbrowser.open("file://" + cd + "/nz_map")
k = 0
