from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

import numpy as np

from .model import Location

import matplotlib.pyplot as plt

import cartopy.crs as ccrs

class BaseProblem:
    """
    Assumes one depo, set of locations to be served on one day

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
            return self.nodes[from_node].time_to(self.nodes[to_node])
        transit_callback_index = self.routing.RegisterTransitCallback(distance_callback)
        dimension_name = 'Distance'
        self.routing.AddDimension(
            transit_callback_index,
            0,
            10000, # 8 hour working day
            True,
            dimension_name,
        )
        distance_dimension = self.routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(100)
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
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
            max_route_time = 0
            for vehicle_id in range(self.num_vehicles):
                index = self.routing.Start(vehicle_id)
                plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
                route_time = 0
                while not self.routing.IsEnd(index):
                    plan_output += ' {} -> '.format(self.manager.IndexToNode(index))
                    previous_index = index
                    index = self.solution.Value(self.routing.NextVar(index))
                    route_time += self.routing.GetArcCostForVehicle(
                        previous_index, index, vehicle_id)
                plan_output += '{}\n'.format(self.manager.IndexToNode(index))
                plan_output += 'Distance of the route: {}m\n'.format(route_time)
                print(plan_output)
                max_route_time = max(route_time, max_route_time)
            print('Maximum of the route distances: {}m'.format(max_route_time))
        else:
            print("Not Solved")
        print(self.objective)


    def plot_solution(self):
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.stock_img()
        lon = [loc.lon for loc in self.nodes]
        lat = [loc.lat for loc in self.nodes]

        ax.scatter(lon, lat, transform=ccrs.Geodetic())

        plt.show()

        
