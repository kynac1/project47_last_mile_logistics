from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

import numpy as np

import matplotlib.pyplot as plt

try:
    import osmnx as ox
except:
    pass
import networkx as nx


class ORToolsRouting:
    """
    Assumes one depo, set of locations to be served on one day

    Attributes
    ----------
    locs : int
        The number of locations. Includes depos.
    
    depo : int, optional
        The location to use as the depo. Ignored if starts and ends are set. Defaults to location 0.

    starts : list[int], optional
        A list of starting locations for vehicles. Length must be equal to number of vehicles
    ends : list[int], optional
        A list of finish locations for vehicles. Length must be equal to number of vehicles
    """

    def __init__(self, locs: int, num_vehicles=int, depo=0, starts=None, ends=None):
        self.locs = locs
        self.num_vehicles = num_vehicles
        self.depo = depo
        self.starts = starts
        self.ends = ends

        self.manager = None
        self.routing = None
        self.solution = None
        self.search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        self._solution = None
        self.objective = np.Inf

    def create_model(self):
        """ Creates required ortools objects and stores them in class variables.

        This could potentially be moved into the init function, but I'm concerned we may need to reset these objects,
        which is easier if this is wrapped separately.
        """
        if self.starts is None and self.ends is None:
            self.manager = pywrapcp.RoutingIndexManager(
                self.locs, self.num_vehicles, self.depo
            )
        else:
            assert len(self.starts) == len(self.ends) == self.num_vehicles
            self.manager = pywrapcp.RoutingIndexManager(
                self.locs, self.num_vehicles, self.starts, self.ends
            )

        self.routing = pywrapcp.RoutingModel(self.manager)

    def add_dimension(
        self,
        distance_matrix: np.array,
        slack_max: int,
        capactity: int,
        fix_start_cumul_to_zero: bool,
        name: str,
    ):
        """ This pattern is reused a lot, so I've rewritten a wrapper to make it less messy

        https://developers.google.com/optimization/reference/python/constraint_solver/pywrapcp#adddimension

        Parameters
        ----------
        distance_matrix : array-like
            A 2D np.array of integers that has the distances from all nodes to all other nodes.
        
        Otherwise, see the above link. 

        Warning
        -------
        The distances must be integer. You can provide non-integral values, but ortools converts them to integer somehow, which will
        mean results are wrong, especially when the distances are small.

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

        def callback(from_index: int, to_index: int):
            """ This is the callback passed to the routing solver. It is a closure over the data in the distance matrix.
            """
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            return distance_matrix[from_node, to_node]

        # Various ortools specific logic. All of this is required, and must be done in this order.
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

    def add_time_windows(
        self,
        time_matrix: np.array,
        time_windows: dict,
        slack_max: int,
        capacity: int,
        fix_start_cumul_to_zero: bool,
        name: str,
    ):
        """ Adds a time windowed constraint

        https://developers.google.com/optimization/reference/python/constraint_solver/pywrapcp#adddimension

        Parameters
        ----------
        time_matrix : array-like
            A 2D np.array of integers that has the times from all nodes to all other nodes.
        time_windows : dict
            A dict that is indexed by the location number, returning an 1d array-like structure with [start_window, end_window]
        
        Otherwise, see the above link. 

        Warning
        -------
        The times must be integer. You can provide non-integral values, but ortools converts them to integer somehow, which will
        mean results are wrong, especially when the times are small.

        See Also
        --------
        add_dimension
        """
        if self.routing is None:
            self.create_model()

        def callback(from_index: int, to_index: int):
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            return time_matrix[from_node, to_node]

        # Same sort of logic as for add_dimension
        transit_callback_index = self.routing.RegisterTransitCallback(callback)
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        time = name
        self.routing.AddDimension(
            transit_callback_index,
            slack_max,
            int(capacity),
            fix_start_cumul_to_zero,
            time,
        )
        time_dimension = self.routing.GetDimensionOrDie(time)

        # Place time windows on dimension
        for location_idx in range(self.locs):
            if (self.starts is None or self.ends is None) or (
                location_idx not in self.starts and location_idx not in self.ends
            ):
                # Ends have no cumulative variable to set a range on.
                index = self.manager.NodeToIndex(location_idx)
                time_dimension.CumulVar(index).SetRange(
                    int(time_windows[location_idx][0]),
                    int(time_windows[location_idx][1]),
                )

        # This code was in the example, seems to be for minimizing time. Don't think it's needed here though.
        for i in range(self.num_vehicles):
            self.routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(self.routing.Start(i))
            )
            self.routing.AddVariableMinimizedByFinalizer(
                time_dimension.CumulVar(self.routing.End(i))
            )

        return time_dimension, transit_callback_index

    def add_disjunction(self, node, penalty):
        """ Allows the solver to drop the node

        Parameters
        ----------
        node : int
            The node number to allow droppint
        penalty : int
            The cost for dropping the node
        """
        self.routing.AddDisjunction([self.manager.NodeToIndex(node)], penalty)

    def add_option(self, nodes: list, penalty):
        self.routing.AddDisjunction(
            list(map(self.manager.NodeToIndex, nodes)), penalty, 1
        )

    def solve(self, tlim=10, log=True):
        """ Solves the route. If the solution has a better objective, this saves the solution.
        """

        if self.routing is None:
            self.create_model()

        self.search_parameters.time_limit.seconds = tlim
        self.search_parameters.log_search = log

        sol = self.routing.SolveWithParameters(self.search_parameters)

        if self.routing.status() == 1:
            v = sol.ObjectiveValue()
            if log:
                print(
                    f"Solved: {self.routing.status()} ({SearchStatus[self.routing.status()]})"
                )
                print(f"Objective: {v}")
            if v < self.objective:
                self._solution = sol
                self.objective = v
                return self.get_solution()

        return None

    def get_solution(self, ortools_sol=None):
        """ Returns the solution in a format independent of ortools.
        """

        if ortools_sol is None:
            ortools_sol = self._solution

        if self.routing is not None and ortools_sol is not None:
            routes = []
            for vehicle_id in range(self.num_vehicles):
                route = []
                index = self.routing.Start(vehicle_id)
                while not self.routing.IsEnd(index):
                    loc = self.manager.IndexToNode(index)
                    route.append(loc)
                    index = self._solution.Value(self.routing.NextVar(index))
                loc = self.manager.IndexToNode(index)
                route.append(loc)

                routes.append(route)

            # Construct, save and return solution object
            self.solution = RoutingSolution(routes)
            return self.solution

        # None if routing not completed at all
        return None


SearchStatus = {
    0: "ROUTING_NOT_SOLVED: Problem not solved yet.",
    1: "ROUTING_SUCCESS: Problem solved successfully.",
    2: "ROUTING_FAIL: No solution found to the problem.",
    3: "ROUTING_FAIL_TIMEOUT: Time limit reached before finding a solution.",
    4: "ROUTING_INVALID: Model, model parameters, or flags are not valid.",
}


class RoutingSolution:
    """ Independent solution object

    The idea is that this can be produced by the routing model. This can then be sent to a simulation function.
    If we try new methods, they should still return routes in this format. This means that we can still use the
    simulation functions with new methods.
    """

    def __init__(self, routes: list):
        self.routes = routes

    def __str__(self):
        """ String representation of the solution for printing
        """
        s = "Routing solution:\n"
        for route in self.routes:
            s += "->".join(str(loc) for loc in route) + "\n"
        return s

    def plot(self, weight_matrix=None, positions=None):
        G = nx.DiGraph()

        for route in self.routes:
            for i in range(len(route) - 1):
                G.add_edge(route[i], route[i + 1])

        if positions:
            pos = {i: positions[i] for i in range(len(positions))}
        else:
            pos = nx.spring_layout()

        nx.draw(G, pos, with_labels=True)

        if weight_matrix is not None:
            labels = {e: str(weight_matrix[e[0], e[1]]) for e in G.edges}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    def plot_osm(self, latlons, G, nodes=None):

        if nodes is None:
            nodes = [ox.get_nearest_node(G, latlon) for latlon in latlons]

        osm_routes = []
        colorlist = []
        nodecolorlist = []
        for route in self.routes:
            color = np.random.rand(3)
            for i in range(len(route) - 1):
                orig_node = nodes[route[i]]
                dest_node = nodes[route[i + 1]]
                path = nx.shortest_path(G, orig_node, dest_node, weight="length")
                osm_routes.append(path)
                colorlist += [color] * (len(path) - 1)
                nodecolorlist += [color] * 2

        return ox.plot_graph_routes(
            G, osm_routes, route_color=colorlist, orig_dest_node_color=nodecolorlist
        )

