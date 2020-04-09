from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

import numpy as np

from project47.model import Location, Vehicle

import matplotlib.pyplot as plt

import cartopy.crs as ccrs

class BaseProblem:
    """
    Assumes one depo, set of locations to be served on one day

    Attributes
    ----------
    depo : Location
        The location the vehicles are based
    locs : list[Location]

    """

    def __init__(self, depo:Location, locs:list, num_vehicles:int):
        self.depo = depo
        self.locs = locs
        self.nodes = locs + [depo]
        self.depo_node = len(self.locs)
        self.num_vehicles = num_vehicles

        self.manager = None
        self.routing = None
        self.solution = None
        self.search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        self._solution = None
        self.objective = np.Inf

    def create_model(self):
        """
        """
        self.manager = pywrapcp.RoutingIndexManager(
            len(self.nodes),
            self.num_vehicles,
            self.depo_node
        )
        self.routing = pywrapcp.RoutingModel(self.manager)
    
    def add_dimension(self, f, slack_max:int, capactity:int, fix_start_cumul_to_zero:bool, name:str):
        """

        This pattern is reused a lot, so I've rewritten a wrapper to make it less messy

        https://developers.google.com/optimization/reference/python/constraint_solver/pywrapcp#adddimension

        Parameters
        ----------
        f : Function
            Some function that takes two locations and returns a scalar value
        Otherwise, see the above link

        Returns
        -------
        dimension : RoutingDimension
            An ortools dimension object. We can set certain types of objectives on this, mostly span costs (cost for having routes with different lengths)
        callback_index : int
            The index of the callback in the routing model.
            This is used with self.routing to add some other types of objectives (more classical distance measures)
        """
        if self.routing is None:
            self.create_model()
        def callback(from_index:int, to_index:int):
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            return self.nodes[from_node].distance_to(self.nodes[to_node])
        transit_callback_index = self.routing.RegisterTransitCallback(callback)
        dimension_name = name
        self.routing.AddDimension(
            transit_callback_index,
            slack_max,
            capactity,
            fix_start_cumul_to_zero,
            dimension_name,
        )
        dimension = self.routing.GetDimensionOrDie(dimension_name)
        return dimension, transit_callback_index
    
    def solve(self, tlim=10, log=True):
        if self.routing is None:
            self.create_model()
        self.search_parameters.time_limit.seconds = tlim
        self.search_parameters.log_search = log
        sol = self.routing.SolveWithParameters(self.search_parameters)
        if self.routing.status() == 1:
            v = sol.ObjectiveValue()
            if log:
                print(f"Solved: {self.routing.status()}")
                print(f"Objective: {v}")
            if v < self.objective:
                self._solution = sol
                self.objective = v
                return self.get_solution()

        return None

    def get_solution(self, ortools_sol=None):
        """ Returns the solution in a format independent of ortools.
        """
        if ortools_sol is None: ortools_sol = self._solution
        routes = []
        vehicles = []
        if self.routing is not None and ortools_sol is not None:
            for vehicle_id in range(self.num_vehicles):
                vehicles.append(self.get_vehicle(vehicle_id))
                route = []
                index = self.routing.Start(vehicle_id)
                while not self.routing.IsEnd(index):
                    loc = self.nodes[self.manager.IndexToNode(index)]
                    route.append(loc)
                    index = self._solution.Value(self.routing.NextVar(index))
                routes.append(route)
            self.solution = Solution(routes, vehicles)
            return self.solution
        return None

    def get_vehicle(self,n:int):
        """ Construct an object for the nth vehicle

        Parameters
        ----------
        n
            Integer from 0..self.num_vehicles
        """
        return Vehicle()

class TimeWindows:
    def __init__(self, depo:Location, locs:list, num_vehicles:int):
        self.depo = depo
        self.locs = locs
        self.nodes = locs + [depo]
        self.depo_node = len(self.locs)
        self.num_vehicles = num_vehicles

        self.manager = None
        self.routing = None
        self.solution = None
        self.objective = np.Inf

    def create_model(self):
        self.manager = pywrapcp.RoutingIndexManager(
            len(self.nodes),
            self.num_vehicles,
            self.depo_node
        )
        self.routing = pywrapcp.RoutingModel(self.manager)

        def distance_callback(from_index:int, to_index:int):
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            return self.nodes[from_node].distance_to(self.nodes[to_node])

        transit_callback_index = self.routing.RegisterTransitCallback(distance_callback)
        
        dimension_name = 'Distance'
        self.routing.AddDimension(
            transit_callback_index,
            0,
            10000,
            True,
            dimension_name,
        )
        distance_dimension = self.routing.GetDimensionOrDie(dimension_name)
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        def time_callback(from_index:int, to_index:int):
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            return self.nodes[from_node].time_to(self.nodes[to_node])

        transit_callback_index = self.routing.RegisterTransitCallback(time_callback)
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        time = 'Time'
        self.routing.AddDimension(
            transit_callback_index,
            0,
            10000,
            False,
            time,
        )
        time_dimension = self.routing.GetDimensionOrDie(time)
        for location_idx, request in enumerate(self.locs):
            index = self.manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(request.time_window[0], request.time_window[1])
        for i in range(self.num_vehicles):
            self.routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(self.routing.Start(i))
            )
            self.routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(self.routing.End(i))
            )


    def solve(self):
        if self.routing is None:
            self.create_model()
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH
        )
        search_parameters.time_limit.seconds = 2
        search_parameters.log_search = True
        sol = self.routing.SolveWithParameters(search_parameters)
        if self.routing.status() == 1:
            print("Solved")
            v = sol.ObjectiveValue()
            if v < self.objective:
                self.solution = sol
                self.objective = v
        else:
            print(self.routing.status())

    def print_solution(self):
        """Prints solution on console."""
        if self.routing is not None and self.solution is not None:
            time_dimension = self.routing.GetDimensionOrDie('Time')
            total_time = 0
            for vehicle_id in range(self.num_vehicles):
                index = self.routing.Start(vehicle_id)
                plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
                while not self.routing.IsEnd(index):
                    time_var = time_dimension.CumulVar(index)
                    plan_output += '{0} Time({1},{2}) -> '.format(
                        self.manager.IndexToNode(index), self.solution.Min(time_var),
                        self.solution.Max(time_var))
                    index = self.solution.Value(self.routing.NextVar(index))
                time_var = time_dimension.CumulVar(index)
                plan_output += '{0} Time({1},{2})\n'.format(self.manager.IndexToNode(index),
                                                            self.solution.Min(time_var),
                                                            self.solution.Max(time_var))
                plan_output += 'Time of the route: {}min\n'.format(
                    self.solution.Min(time_var))
                print(plan_output)
                total_time += self.solution.Min(time_var)
            print('Total time of all routes: {}min'.format(total_time))

    def plot_solution(self):
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.stock_img()
        lon = [loc.lon for loc in self.nodes]
        lat = [loc.lat for loc in self.nodes]

        ax.scatter(lon, lat, transform=ccrs.Geodetic())

        if self.routing is not None and self.solution is not None:
            for vehicle_id in range(self.num_vehicles):
                color = np.random.rand(3)
                index = self.routing.Start(vehicle_id)
                while not self.routing.IsEnd(index):
                    a = self.nodes[self.manager.IndexToNode(index)]
                    index = self.solution.Value(self.routing.NextVar(index))
                    b = self.nodes[self.manager.IndexToNode(index)]
                    ax.plot([a.lon,b.lon],[a.lat,b.lat], color=color, transform=ccrs.Geodetic())

        plt.show()

class Solution:
    """ Independent solution object

    The idea is that this can be produced by the routing model.

    Simulations could then take this initial solution as an input.
    """
    def __init__(self, routes:list, vehicles=[], default_vehicle=Vehicle()):
        self.routes = routes
        self.vehicles = vehicles
        if not self.vehicles:
            self.vehicles = [default_vehicle]*len(routes)
        self.calculate_predictions()
    
    def calculate_predictions(self):
        """ Calculates predicted time and distance between points.

        Could generalise this to other quantities, like capacity, but not sure there's much point right now.
        """
        self.predicted_time = [np.zeros(len(route)) for route in self.routes]
        self.predicted_distance = [np.zeros(len(route)) for route in self.routes]
        for i,route in enumerate(self.routes):
            for j,loc in enumerate(route):
                if j == 0: 
                    self.predicted_distance[i][j] = 0
                    self.predicted_time[i][j] = 0
                else: 
                    self.predicted_distance[i][j] = self.predicted_distance[i][j-1] + route[j-1].distance_to(loc)
                    self.predicted_time[i][j] = self.predicted_time[i][j-1] + route[j-1].time_to(loc)
    
    def __str__(self):
        """ String representation of the solution for printing
        """
        return "Solution: " + str(self.routes)

    def plot(self):
        """ Plots the locations and routes on a map

        Not sure of the resolution needed, and zooming in can cause issues with images taking too long.
        I'll fix this once we start working with more realistic locations.
        """
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.stock_img()
        lon = [loc.lon for route in self.routes for loc in route]
        lat = [loc.lat for route in self.routes for loc in route ]

        ax.scatter(lon, lat, transform=ccrs.Geodetic())

        for route in self.routes:
            lon = [loc.lon for loc in route] + [route[0].lon]
            lat = [loc.lat for loc in route] + [route[0].lat]
            ax.plot(lon, lat, transform=ccrs.Geodetic(), color=np.random.rand(3))

        plt.show()
            
        

        
     
