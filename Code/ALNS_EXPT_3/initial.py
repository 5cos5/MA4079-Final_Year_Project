from read_data_expt2 import Input, InitialData, InputParameters
from model_check import route_check_single, cust_no_index
from trip_generator import create_trips_single
import numpy as np
import copy

def initial_greedy_insertion_distance(data, initial_data, parameters):
    demands = np.array(data.demands)
    indexes = data.indexes
    # cust_no = np.array(data.cust_no)
    distances = np.array(data.distances)
    MAX_CAPACITY = tuple(parameters.MAX_CAPACITY)
    # sum_demands = -np.sort(-np.sum(demands, axis =1))
    routes = []
    for k in range(parameters.N_VEHICLES):
        # index = np.where(demands == sum_demands[k])[0][0]
        # routes.append([0]) #,cust_no[index]])
        routes.append([cust_no_index(cust,data) for cust in initial_data.initial_route[k]])
    # print('initial_route', routes)
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
    ''''
    ## Select furthurest customer if we are starting at the start else continue
    for k,route in enumerate(routes):
        while len(route) == 1:
            i =route[-1]
            temp_route = copy.deepcopy(route)
            print(i)
            unserved_distances = [distances[i,j] for j in unserved_cust]
            max_index = np.argmax(unserved_distances)
            temp_route.append(unserved_cust[max_index])

            if route_check_single(temp_route,data,initial_data,parameters,k):
                route = temp_route
                unserved_cust.remove(unserved_cust[max_index])
                print(route)
        greedy_routes.append(route)'''

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
 
            # temp_route.append(0)
            # print('temp_route b4 create trips', temp_route)
            temp_route = create_trips_single(temp_route,data,initial_data,parameters,k)
            # print('temp_route', temp_route)
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