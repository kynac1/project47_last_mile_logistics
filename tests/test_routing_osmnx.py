from project47.routing import ORToolsRouting
import numpy as np
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

def test_christchurch_example():
    latlons = [
        (-43.5111688, 172.7319266),
        (-43.5499101, 172.63913),
        (-43.547876 , 172.6473834),
        (-43.5295274, 172.5866324),
        (-43.549314 , 172.622618),
        (-43.5194318, 172.6443871),
        (-43.5262727, 172.5677181),
        (-43.5251111, 172.5768752),
        (-43.5308677, 172.6454779),
        (-43.526729 , 172.6100037),
        (-43.5334894, 172.6601199),
        (-43.5345317, 172.6593844),
        (-43.5199411, 172.7308814),
        (-43.5280717, 172.6314508),
        (-43.5217694, 172.6215047),
        (-43.5761387, 172.7624258),
        (-43.5220461, 172.6280757),
        (-43.4779506, 172.5798551),
        (-43.5055327, 172.619338),
        (-43.5255113, 172.6468588),
        (-43.5296929, 172.6531345),
        (-43.5163431, 172.5716451),
        (-43.4955977, 172.7070804),
        (-43.526729 , 172.6100037),
        (-43.5589434, 172.6344832),
        (-43.5055445, 172.6365762),
        (-43.5170442, 172.6462857),
        (-43.5252464, 172.6293588),
        (-43.5184613, 172.6486995),
        (-43.5194318, 172.6443871),
        (-43.4938774, 172.7122268),
        (-43.5708894, 172.7462063),
        (-43.5132977, 172.5945826),
        (-43.5328785, 172.6514435),
        (-43.5232292, 172.5464753),
        (-43.552104 , 172.5654115),
        (-43.5278681, 172.5847619),
        (-43.5141478, 172.5387021),
        (-43.5509728, 172.642015),
        (-43.5208447, 172.6234537),
        (-43.5679401, 172.6609946),
        (-43.537706 , 172.663747),
        (-43.5374932, 172.6056399),
        (-43.5427405, 172.6636662),
        (-43.5159515, 172.6303761)
    ]
    locs = len(latlons)
    place = 'Christchurch, NZ'
    G = ox.graph_from_place(place, network_type='drive')
    
    nodes = [ox.get_nearest_node(G, latlon) for latlon in latlons]

    # Build the distance matrix. This should be easily replaced with google maps api's if needed.
    distance_matrix = np.zeros((locs,locs))
    for i in range(locs):
        for j in range(locs):
            distance_matrix[i,j] = nx.shortest_path_length(G, nodes[i], nodes[j], weight='length')

    prob = ORToolsRouting(locs, 10) # First location is implicitly the depo

    dist_dim, dist_ind = prob.add_dimension(distance_matrix, 0, 50000, True, "Distance")
    #dist_dim.SetGlobalSpanCostCoefficient(100)
    prob.routing.SetArcCostEvaluatorOfAllVehicles(dist_ind)
    
    solution = prob.solve()

    if solution:
        print(solution)
        fig, ax = solution.plot_osm(latlons, G)
        plt.show()

if __name__ == "__main__":
    test_christchurch_example()