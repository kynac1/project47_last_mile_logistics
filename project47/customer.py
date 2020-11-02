import numpy as np


class Customer:
    """Customer object

    An object representing a single location for a package to be delivered to.

    Attributes
    ----------
    lat : number
        latitude coordinate
    lon : number
        longitude cordinate
    responsiveness : 0..1
        Fixed responsiveness, the probability of a successful delivery given the customer is present
    call_responsiveness : 0..1
        The probability of a call being successful at any point in time
    presence : array-like
        An array of binary values, controlling whether the customer is present at a given time
    presence_interval : int
        The width of each entry in the presence array, in time
    rg : np.random.Generator
        A random bitstream to draw random numbers from
    alternates : list[Customer]
        Alternate customers that the same package could be delivered to instead
    """

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
        """Adds an alternative delivery customer

        Appends the new customer to the alternates list.
        Recursively does this for every attached customer. So this framework can accomodate arbitrary numbers of alternative delivery locations.

        Parameters
        ----------
        c : Customer
        """
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

        Parameters
        ----------
        time : int
            The time the visit takes place

        Returns
        -------
        bool
            True if delivery was successful
        """
        indp = int(time) // self.presence_interval
        if indp >= len(self.presence):
            return False
        return bool(self.presence[indp]) and self.rg.random() < self.responsiveness

    def get_time_window(self, options=[[0, 1]]):
        """Finds the best time window for a customer

        Chooses the best time window to maximise the sum of the presence list within it.

        Attributes
        ----------
        options : list[list[int]]
            Time-windows are a pair of start-end times. Options is therefore a list of these pairs.

        Returns
        -------
        list[int]
            The best option from the given options
        """
        best = 0
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
        """Calls the customer with an arrival time, to see if they will be home

        Parameters
        ----------
        arrival_time : int
            The time we expect the delivery to take place

        Returns
        -------
        bool
            True if customer responds saying delivery will be successful
        """
        if self.rg.random() < self.call_responsiveness:
            return self.visit(arrival_time)
        else:
            return False

    def call_ahead_tw(self, arrival_time, options=[]):
        """Experimental function to call ahead and get a new, more accurate time-window for delivery"""
        if self.rg.random() < self.call_responsiveness:
            return self.get_time_window(options)
        else:
            return [0, 1]  # Minimum width time window. [0,0] may break ortools.


def markov_presence(n, prob: float, rg: np.random.Generator):
    """Simulates a symmetric markov chain, to generate a presence array for the customer."""
    presence = np.zeros(n, dtype=bool)
    presence[0] = True  # Start in state with 50% chance
    for i in range(n - 1):
        if rg.random() < prob:
            presence[i + 1] = not presence[i]
        else:
            presence[i + 1] = presence[i]

    return presence
