from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

import numpy as np

from project47.model import Location

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
            40,
            True,
            dimension_name,
        )
        distance_dimension = self.routing.GetDimensionOrDie(dimension_name)
        #distance_dimension.SetGlobalSpanCostCoefficient(100)
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
            max_route_distance = 0
            for vehicle_id in range(self.num_vehicles):
                index = self.routing.Start(vehicle_id)
                plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
                route_distance = 0
                while not self.routing.IsEnd(index):
                    plan_output += ' {} -> '.format(self.manager.IndexToNode(index))
                    previous_index = index
                    index = self.solution.Value(self.routing.NextVar(index))
                    route_distance += self.routing.GetArcCostForVehicle(
                        previous_index, index, vehicle_id)
                plan_output += '{}\n'.format(self.manager.IndexToNode(index))
                plan_output += 'Distance of the route: {}m\n'.format(route_distance)
                print(plan_output)
                max_route_distance = max(route_distance, max_route_distance)
            print('Maximum of the route distances: {}m'.format(max_route_distance))
        else:
            print("Not Solved")
        print(self.objective)

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

        
     
