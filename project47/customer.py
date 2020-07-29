import numpy as np

class Customer:
    def __init__(self, lat, lon, responsiveness, call_responsiveness, presence):
        self.lat = lat
        self.lon = lon
        self.responsiveness = responsiveness
        self.call_responsiveness = call_responsiveness
        self.presence = presence

    def visit(self, time):
        """ Called to determine whether a customer successfully recieves a package.

        Returns True on a successful delivery
        """
        for tw in self.presence:
            if time < tw[1] and time > tw[0]:
                return np.random.rand() < self.responsiveness
        return False
            