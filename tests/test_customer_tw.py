import numpy as np
import random
from project47.data import *
from project47.routing import *
from project47.simulation import *
# from project47.multiday_simulation import *
# from project47.customer import Customer
from functools import reduce
# from numpy.random import Generator, PCG64

# seed=None
# lat = np.zeros(10)
# # rg = Generator(PCG64(seed))
# num_presence = 8
# n = random.randint(0,2**num_presence-1)
# presence = list(map(int,'{0:08b}'.format(n)))
# # rd = rg.integers(low=0, high=len(sample_df)-1, size=n)
# print(presence)
# # print(bin(256))

# day_start = 0
# day_end = 28800
# num_time_windows = 4

# presence_per_tw = int(num_presence/num_time_windows)
# temp = [sum(presence[i:i+presence_per_tw]) for i in range(0, len(presence), presence_per_tw)]
# print(temp)
# max_ind = temp.index(max(temp))

def sample_generator():

        day_start = 0
        day_end = 28800
        num_time_windows = 4
        interval = (day_end - day_start) / num_time_windows
        num_presence = 8
        presence_per_tw = int(num_presence/num_time_windows)
        # lat, lon = get_sample(5, rg, cd, sample_df, CHC_df, CHC_sub, CHC_sub_dict, save=False)
        presence_list = []
        lat = np.zeros(10)
        time_windows = np.zeros((len(lat),2))

        for i in range(len(lat)):
            # get a random decimal number between 0-255
            # rd = rg.integers(low=0, high=2**num_presence-1, size=1) # a list of random numbers
            rd = random.randint(0,2**num_presence-1)
            # get a list of binary number representing the presence for each presence interval
            presence = list(map(int,'{0:08b}'.format(rd)))
            presence_list.append(presence)
            # find the time window with highest presence
            sum_per_tw = [sum(presence[i:i+presence_per_tw]) for i in range(0, len(presence), presence_per_tw)]
            max_ind = sum_per_tw.index(max(sum_per_tw))

            time_windows[i,0] = interval * max_ind
            time_windows[i,1] = interval * (max_ind+1)
        # customers = [Customer(lat,lon, 0.9, 0.9, [time_windows[i,:]]) for i in range(len(lat))]

        return time_windows
        # return customers, time_windows



if __name__ == "__main__":
    seed=123456789
    # rg = Generator(PCG64(seed))
    sample_generator()


# rd = rg.integers(low=0, high=256, size=n) # a list of random numbers