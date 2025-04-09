from model_check import cust_no_index
from read_data_expt2 import Input, InitialData, InputParameters
import numpy as np
import logging

def remove_repeated_depot(routes):
    cleaned_routes = []
    for route in routes:
        temp = [route[i] for i in range(len(route)) if i == 0 or route[i] != route[i-1]]
        cleaned_routes.append(temp)
    return cleaned_routes

def node_no_index(node, data):
    '''
    takes in node number and returns the corresponding customer number
    '''
    return data.cust_no[node]

def create_trips_single(route,data,initial_data,parameters,k):
    '''
    takes in a single route and splits it up into trips, such that each trip is feasible loading wise
    '''
    MAX_CAPACITY = tuple(parameters.MAX_CAPACITY)
    route_trip = [cust_no_index(cust,data) for cust in initial_data.initial_route[k]]
    load = np.array(initial_data.initial_loading[k])
    for a,node in enumerate(route[1:]):
        if node == 0:
            load = MAX_CAPACITY
            route_trip.append(0)
        else:
            #calculate loading
            service_load = np.array(data.demands[node])
            if (service_load>MAX_CAPACITY).any(): #service load larger than vehicle cap
                raise Exception('Unable to service customer as its demands exceed vehicle load')
                #skip this node and ignore
                # continue
                # logging.error("Load at node exceeds vehicle capacity")
            else:
                load -= service_load
                if (load<parameters.EMPTY_CAPACITY).any(): # load in veh is too little
                    load = MAX_CAPACITY
                    route_trip.append(0)
                    route_trip.append(node)
                else:
                    route_trip.append(node)
    if route_trip[-1] != 0: #trip does not end at depot
        route_trip.append(0)
    # route_trip = [node_no_index(node,data) for node in route_trip]

    return route_trip

def create_trips(routes, data, initial_data, parameters):
    '''
    takes in a route and splits it up into trips, such that each trip is feasible loading wise
    '''
    route_trips = []
    for k, route_veh in enumerate(routes):
        # route_veh = [cust_no_index(cust,data) for cust in route_veh]
        MAX_CAPACITY = tuple(parameters.MAX_CAPACITY)
        route_trip = [cust_no_index(cust,data) for cust in initial_data.initial_route[k]]
        load = np.array(initial_data.initial_loading[k])
        for a,node in enumerate(route_veh[1:]):
            if node == 0:
                load = MAX_CAPACITY
                route_trip.append(0)
            else:
                #calculate loading
                service_load = np.array(data.demands[node])
                if (service_load>MAX_CAPACITY).any(): #service load larger than vehicle cap
                    raise Exception('Unable to service customer as its demands exceed vehicle load')
                    #skip this node and ignore
                    # continue
                    # logging.error("Load at node exceeds vehicle capacity")
                else:
                    load -= service_load
                    if (load<parameters.EMPTY_CAPACITY).any(): # load in veh is too little
                        load = MAX_CAPACITY
                        route_trip.append(0)
                        route_trip.append(node)
                    else:
                        route_trip.append(node)
        if route_trip[-1] != 0: #trip does not end at depot
            route_trip.append(0)
        # route_trip = [node_no_index(node,data) for node in route_trip]
        route_trips.append(route_trip)
    return route_trips



def remove_depots(route):
    temp =[]
    for i, cust in enumerate(route):
        if cust == 1:
            continue
        else:
            temp.append(cust)
    return temp

def remove_trips(routes):
    temp_routes = []
    for route in routes:
        temp = remove_depots(route)
        temp = np.hstack((1,temp,1))
        temp_routes.append(list(temp))
    return temp_routes

if __name__ == '__main__':
    N = 100
    FILE = 'R201_ordered'
    CAPACITY = 50
    DEMAND_TYPES = 5
    VEHICLE_COMPARMENTS = 5
    N_VEHICLES = 3
    MAX_CAPACITY = np.array([CAPACITY]*DEMAND_TYPES)
    EMPTY_CAPACITY = np.array([0]*DEMAND_TYPES)
    data = Input.load_csv(FILE= FILE, N= N)
    parameters = InputParameters(N_VEHICLES=N_VEHICLES,DEMAND_TYPES=DEMAND_TYPES,VEHICLE_COMPARMENTS=VEHICLE_COMPARMENTS,
                                  DEPOT_LOADING_SCALAR=1,MAX_CAPACITY=MAX_CAPACITY,EMPTY_CAPACITY=EMPTY_CAPACITY)
    route1_cust_no = [1,36]
    route2_cust_no = [1,75]
    route3_cust_no = [1,14]

    time_start_1 = 644.96
    time_start_2 = 769.4
    time_start_3 = 786.69

    loading1 = [CAPACITY,CAPACITY,CAPACITY,CAPACITY,CAPACITY]
    loading2 = [CAPACITY,CAPACITY,CAPACITY,CAPACITY,CAPACITY]
    loading3 = [CAPACITY,CAPACITY,CAPACITY,CAPACITY,CAPACITY]

    initial_data = InitialData(
        initial_route=[route1_cust_no, route2_cust_no, route3_cust_no],
        initial_loading= [loading1,loading2,loading3],
        initial_time= [time_start_1,time_start_2,time_start_3]
    )
    route1 = [1, 40, 96, 93, 64, 48, 1, 60, 28, 29, 84, 6, 43, 3, 1, 32, 1]
    route2 = [1, 1]
    route3 = [1, 93, 64, 48, 1, 60, 28, 29, 84, 6, 43, 3, 1, 32, 73, 1]

    routes = [[1, 20, 12, 8, 9, 21, 4], [1, 1], [1, 13, 19, 10, 7, 17, 1, 11, 18, 6, 1, 14, 1]]
    print (create_trips(routes,data,initial_data,parameters))