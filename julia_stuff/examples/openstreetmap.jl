using OpenStreetMapX

m = get_map_data("resources/map.osm") # Loads map data

a = LLA(-43.524851, 172.569875)
b = LLA(-43.508310, 172.575100)

a_node = point_to_nodes(a, m)
b_node = point_to_nodes(b, m)

println("Points: $a, $b")

println("Straight line distance between points: $(distance(ECEF(a), ECEF(b)))") 

# This gets the shortest route along roads, with an estimated distance and time
r = shortest_route(m, a_node, b_node)
println("Shortest Path along roads")
println("Distance: $(r[2])")
println("Time: $(r[3])")

r = fastest_route(m, a_node, b_node)
println("Fastest Path along roads")
println("Distance: $(r[2])")
println("Time: $(r[3])")

