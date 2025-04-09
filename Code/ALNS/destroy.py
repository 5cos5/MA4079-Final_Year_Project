import numpy as np
from trip_generator import  remove_trips
from model_check import cust_no_index
from functools import cache

def find_cust(routes,cust_to_find):
    for k, route in enumerate(routes):
            for i, cust in enumerate(route):
                if cust ==cust_to_find:
                    return k,i
                
def random_remove_point (routes,data,initial_data,parameters,degree_of_destruction):
    rng = np.random.default_rng()
    # routes = remove_trips(routes)
    def flatten_comprehension(matrix):
        return [item for row in matrix for item in row if item !=0]
    customers = flatten_comprehension(routes)
    for initial_cust in initial_data.initial_route: #remove inital customers, which cannot be touched
        try:
            customers.remove(cust_no_index(initial_cust[0],data))
        except:
            continue
    if len(customers) ==0: #route is empty
        return routes
    for _ in range(degree_of_destruction):
        remove_cust = rng.choice(customers)
        k,i = find_cust(routes,remove_cust)
        del routes[k][i]
        customers.remove(remove_cust)
        '''
        remove_veh = rng.integers(len(routes))
        while len(routes[remove_veh]) <= 1: #roll until we do not get an empty vehicle
            remove_veh = rng.integers(len(routes))
        remove_position = rng.integers(1,len(routes[remove_veh])) #dont remove the first posiiton
        while routes[remove_veh][remove_position] == 0: # if we are removing the depot, just roll again
            remove_veh = rng.integers(len(routes))
            while len(routes[remove_veh]) <= 1: #roll until we do not get an empty vehicle
                remove_veh = rng.integers(len(routes))
            remove_position = rng.integers(1,len(routes[remove_veh])) #dont remove the first posiiton
        del routes[remove_veh][remove_position]'''
    return routes

def greedy_remove_distance(routes,data,initial_data,parameters,degree_of_destruction):
    '''
    ### Removes customers which has the highest distance cost
    '''
    # demands = np.array(data.demands)
    # cust_no = np.array(data.cust_no)
    distances = np.array(data.distances)
    
    for _ in range(degree_of_destruction):
        routes_max_distance = []
        routes_max_index = []
        for route in routes:
            route_distance = []
            for idx, cust in enumerate(route[1:-1]): #ignore first customer
                if cust == 0: # depot
                    route_distance.append(0)
                    continue
                else:
                    # print (idx)
                    prev_cust = route[idx]
                    next_cust = route[idx+2]
                    distance = distances[cust,next_cust]
                    distance += distances[prev_cust,cust]
                    route_distance.append(distance)
            # print (route_distance)
            if len(route_distance) != 0: 
                max_index = np.argmax(np.array(route_distance))
                routes_max_distance.append(route_distance[max_index])
                routes_max_index.append(max_index)
            else: # no customers, route is empty
                routes_max_distance.append(0)
                routes_max_index.append(None)
        veh_index = np.argmax(np.array(routes_max_distance))
        # print("VEHICLE", veh_index)
        if routes_max_index[veh_index] == None: #route is empty
            return routes
        del routes[veh_index][routes_max_index[veh_index]+1]
        # routes = remove_trips(routes)
    return routes
# @cache
def shaw_score(i,j,data,initial_data,parameters,a=1,b=1):
    distance = data.distances[i,j]
    start_time_difference = abs(data.start_time_windows[i] - data.start_time_windows[j])
    end_time_difference = abs(data.end_time_windows[i] - data.end_time_windows[j])
    '''
    loading_temp = np.array(data.demands[i]) + np.array(data.demands[j])
    if (loading_temp>parameters.MAX_CAPACITY).any():
        load_score = '''
    return a*distance + b*(start_time_difference+end_time_difference)

def shaw_removal(routes,data,initial_data,parameters,degree_of_destruction):
    '''
    ### Removes customers which are closely related
    '''
    rng = np.random.default_rng()
    def flatten_comprehension(matrix):
        return [item for row in matrix for item in row if item !=0]
    customers = flatten_comprehension(routes)
    if len(customers) ==0: #route is empty
        return routes
    seed_cust = rng.choice(customers) #select a random cust

    score_dict={}
    for k,route in enumerate(routes):
        for idx, cust in enumerate(route[1:]): #ignore first customer
            if cust == 0: # depot
                continue
            else:
                score = shaw_score(seed_cust,cust,data,initial_data,parameters,a=1,b=1)
                score_dict[(k,idx,cust)] = score
        # print (route_scores)
    sorted_dict = {k: v for k, v in sorted(score_dict.items(), key=lambda item: item[1])}
    # print(sorted_dict)

    sorted_dict_keys = list(sorted_dict.keys())
    # print(sorted_dict_keys)
    for i in range(degree_of_destruction):
        veh_index = sorted_dict_keys[i][0]
        pos_index= sorted_dict_keys[i][1]
        cust = sorted_dict_keys[i][2]

        routes[veh_index].remove(cust) #note:seed customer is also removed
        # print(routes)
    # routes = remove_trips(routes)
    return routes