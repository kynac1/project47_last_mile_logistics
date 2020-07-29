import numpy as np
from project47.routing import *
from numpy.random import Generator, PCG64

def collect_data(day:int, solution:RoutingSolution, distances:list, times:list, futile:np.array, 
                delivered:np.array, arrival_days:list, time_windows:dict):
    data = {}
    key = str(day)

    time_delivered = []
    for d in delivered:
        if d != 0:
            found = False
            for i,route in enumerate(solution.routes):
                for j,node in enumerate(route):
                    if node == d:
                        time_delivered.append(times[i][j])
                        found = True
                        break
                if found:
                    break
            if not found:
                time_delivered.append(-1) # Error. Hopefully negative time is obviously wrong.

    data[key] = {
        "number_of_packages": int(len(arrival_days)-1), # This could potentially be higher than the number of deliveries in the routes
        "number_of_vehicles": int(len(solution.routes)), # is it number of used vehicles or just a total number? ## Either way, I think we don't actually need this; we can get it from other info
        "distances": [int(sum(veh_dists)) for veh_dists in distances],
        "times": [int(sum(veh_times)) for veh_times in times],
        "deliveries_attempted": [int(len(route)-2) for route in solution.routes],#{[1,3,6,0,0]}, # successful deliveries or total deliveries? ## Attempted, so total. -2 for depo at start and end.
        "futile_deliveries": [int(f) for f in futile],
        "delivered_packages": {
            "days_taken": [int(day - arrival_days[i]) for i in delivered if i != 0], #[1,4,2,7,32,2],
            "time_delivered": [int(t) for t in time_delivered],
            "time_window": [[int(time_windows[i][0]),int(time_windows[i][1])] for i in delivered if i != 0] #[[2,4],[2,3],[2,3],[1,4],[1,2]] 
        },
        "undelivered_packages": {
            "days_taken": [int(day - arrival_days[i]) for i in range(len(arrival_days)) if i!=0 or i not in delivered],#[12,54,21,43,21],
            "time_window": [[int(time_windows[i][0]),int(time_windows[i][1])] for i in range(len(arrival_days)) if i!=0 or i not in delivered]#[[3,7],[2,7],[5,9],[1,3],[4,5]] 
        }
    }

    return data


def multiday(depots, sample_generator, dist_and_time, route_optimizer, simulator, n_days, day_start, day_end, seed=None, replications=1):
    """
    Paramters
    ---------
    depots : np.array
        2*n_depots array of longitudes and latitudes of the depots.
        This is set up to support multidepot problems. However, to do this properly we'll need to track which depots
        have which packages. Need to think about this more.
    sample_generator : function
        Takes no inputs, returns two lists, longitudes and latitudes of the packages.
    dist_and_time : function
        Takes longitudes and latitudes, and returns a distance matrix, a time matrix, and a array of time windows.
    route_optimizer : function
        Inputs are the depot numbers, the distance and time matrices, the time windows as a np.array,
        the current day, the day each package arrived, and the number of times each package was futile.
        Outputs are a set of vehicle routes, and a list of packages that were not scheduled.
    simulator : function
        Simulates the deliveries for a set of routes, given the routes, distances, times and time windows.
    n_days : int
        The number of days to simulate.
    day_start : int
        The time for the start of a day
    day_end : int
        The time for the end of a day
    """
    rg = Generator(PCG64(seed))
    data = []
    delivery_lats = depots[0]
    delivery_lons = depots[1]
    n_depots = depots.shape[1]
    delivery_time_windows = np.array([[day_start, day_end] for i in range(n_depots)])
    arrival_days = np.zeros(n_depots)
    futile_count = np.zeros(n_depots)

    latitudes_per_day = []
    longitudes_per_day = []
    time_windows_per_day = []
    
    for day in range(n_days):
        lats, lons, new_time_windows = sample_generator(rg)
        latitudes_per_day.append(lats)
        longitudes_per_day.append(lons)
        time_windows_per_day.append(new_time_windows)

    for day in range(n_days):
        # Generate data 
        lats, lons, new_time_windows = latitudes_per_day[day], longitudes_per_day[day], time_windows_per_day[day]
        delivery_lats = np.append(delivery_lats,lats)
        delivery_lons = np.append(delivery_lons,lons)
        delivery_time_windows = np.vstack((delivery_time_windows,new_time_windows))
        arrival_days = np.append(arrival_days, [day for _ in range(len(lats))])
        futile_count = np.append(futile_count, np.zeros(len(lats)))

        # Get times and distances
        dm,tm = dist_and_time(delivery_lats, delivery_lons)
        if dm is None:
            # We've exceeded the map bounds. Stop here for now, but we should really handle this more gracefully.
            break
        dm = np.array(dm)
        tm = np.array(tm)
        
        # Calulate routes for the day TODO
        routes, unscheduled = route_optimizer(
            [i for i in range(n_depots)], 
            dm, tm, delivery_time_windows, 
            day, arrival_days, futile_count
        )
        if False:
            routes.plot(positions=[(delivery_lats[i], delivery_lons[i]) for i in range(len(delivery_lats))], weights=dm)
        futile_count[[i for i in range(len(delivery_lats)) if i not in unscheduled]] += 1

        for i in range(replications):
            # Simulate behaviour
            distances, times, futile, delivered = simulator(
                routes, dm, tm, delivery_time_windows, rg
            )

            # Data collection to save
            data.append(collect_data(day, routes, distances, times, futile, delivered, arrival_days, delivery_time_windows))

        # Remove delivered packages, using just the last result
        undelivered = np.ones(len(delivery_lats), dtype=bool)
        undelivered[delivered] = False
        undelivered[[i for i in range(n_depots)]] = True
        delivery_lats = delivery_lats[undelivered]
        delivery_lons = delivery_lons[undelivered]
        delivery_time_windows = delivery_time_windows[undelivered]
        arrival_days = arrival_days[undelivered]
        futile_count = futile_count[undelivered]

    return data

