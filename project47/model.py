import numpy as np

import utm


class Location:
    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def time_to(self, b):
        ''' Time between locations

        currently just takes the locations cartesian distance and calculates time travelling at 50 kmph.

        Assumes in utm zone 59, 'G', which should be around Christchurch.
        https://www.maptools.com/tutorials/grid_zone_details
        '''
        return np.linalg.norm(np.array(utm.from_latlon(b.lat, b.lon)[0:2]) - np.array(utm.from_latlon(self.lat, self.lon)[0:2])) * 2e-5


    
    