import numpy as np


class Customer:
    def __init__(
        self,
        lat=0,
        lon=0,
        responsiveness=1,
        call_responsiveness=1,
        presence=np.array([1]),
        presence_interval=28800,
        rg=np.random.Generator(np.random.PCG64(123)),
    ):
        self.lat = lat
        self.lon = lon
        self.responsiveness = responsiveness
        self.call_responsiveness = call_responsiveness
        self.presence = presence
        self.presence_interval = presence_interval
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
        indp = int(time) // self.presence_interval
        return bool(self.presence[indp])

    def get_time_window(self, options=[]):
        pass

    def call_ahead(self, arrival_time):
        if self.rg.random() < self.call_responsiveness:
            return self.visit(arrival_time)
