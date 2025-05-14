from model_check import route_check_single, cust_no_index
from trip_generator import create_trips_single
import numpy as np
import copy

def initial_greedy_insertion_distance(data, initial_data, parameters):
    demands = np.array(data.demands)
    indexes = data.indexes
    distances = np.array(data.distances)
    MAX_CAPACITY = tuple(parameters.MAX_CAPACITY)

    routes = []
    for k in range(parameters.N_VEHICLES):

        routes.append([cust_no_index(cust,data) for cust in initial_data.initial_route[k]])

    greedy_routes = []
    temp = sum(routes,[]) # flatten out routes
    unserved_cust = list(indexes[np.isin(indexes,temp, invert = True)])
    unserved_demands = [demands[j] for j in unserved_cust]
    for idx, unserved_demand in enumerate(unserved_demands):
        if (unserved_demand>MAX_CAPACITY).any():
            # print(unserved_demand)
            unserved_demands.pop(idx)
            unserved_cust.pop(idx)
        if sum(unserved_demand) ==0: #remove the depot
            unserved_demands.pop(idx)
            unserved_cust.pop(idx)

    # return greedy_routes
    for k,route in enumerate(routes):
        FEASIBILITY = True
        while FEASIBILITY:
            i = route[-1]
            unserved_distances = [distances[i,j] for j in unserved_cust]
            if len(unserved_distances) ==0:
                greedy_routes.append(route)
                break
            if len(route) ==1: # Select furthurest customer if we are starting at the start else select cloest customer
                min_index = np.argmax(unserved_distances)
            else:
                min_index = np.argmin(unserved_distances)
            temp_route = copy.deepcopy(route)
            if len(temp_route)>1 and temp_route[-1] == 0: #remove depot at the end, to be added after create trips
                temp_route.pop(-1)
            temp_route.append(unserved_cust[min_index])
 
            temp_route = create_trips_single(temp_route,data,initial_data,parameters,k)
            if route_check_single(temp_route,data,initial_data,parameters,k):
                route = temp_route
                unserved_cust.remove(unserved_cust[min_index])
            else:
                if len(route) <= 1:
                    unserved_cust.remove(unserved_cust[min_index])
                else:
                    FEASIBILITY = False
                    greedy_routes.append(route)
                    break
    return greedy_routes