import numpy as np

class Customer:
    def __init__(self, lat, lon, responsiveness, call_responsiveness, presence, rg):
        self.lat = lat
        self.lon = lon
        self.responsiveness = responsiveness
        self.call_responsiveness = call_responsiveness
        self.presence = presence
        self.alternates = set([self])
        self.rg = rg
    
    def add_alternate(self, c):
        self.alternates.add(c)
        self.alternates |= c.alternates
        c.alternates = self.alternates
        added = True
        while added:
            added = False
            for alternate in self.alternates:
                if self.alternates != alternate.alternates:
                    self.alternates |= alternate.alternates
                    alternate.alternates = self.alternates
                    added = True

    def visit(self, time):
        """ Called to determine whether a customer successfully recieves a package.

        Returns True on a successful delivery
        """
        for tw in self.presence:
            if time <= tw[1] and time >= tw[0]:
                return self.rg.random() < self.responsiveness
        return False
            