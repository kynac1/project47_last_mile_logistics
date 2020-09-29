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
        """Called to determine whether a customer successfully recieves a package.

        Returns True on a successful delivery
        """
        indp = int(time) // self.presence_interval
        return bool(self.presence[indp]) and self.rg.random() < self.responsiveness

    def get_time_window(self, options=[[0, 1]]):
        best = 1
        best_value = 0
        for i, option in enumerate(options):
            value = sum(
                self.presence[
                    int(option[0] // self.presence_interval) : int(
                        option[1] // self.presence_interval
                    )
                ]
            )
            if value > best_value:
                best = i
        return options[best]

    def call_ahead(self, arrival_time):
        if self.rg.random() < self.call_responsiveness:
            return self.visit(arrival_time)
        else:
            return False

    def call_ahead_tw(self, arrival_time, options=[]):
        if self.rg.random() < self.call_responsiveness:
            return self.get_time_window(options)
        else:
            return [0, 1]  # Minimum width time window. [0,0] may break ortools.
