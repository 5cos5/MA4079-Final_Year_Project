import numpy as np
from model_check import trip_split,objective_function,objective_function_fast,route_check_single,route_check_trips,cust_no_index,\
    find_time_slack_newest,check_insertion_linear_newest
# from trip_generator import create_trips_single
# import copy
from collections import defaultdict
import logging
from operator import itemgetter
'''
def random_insert_point(routes,data):
    '''
    ### inserts a customer into a random veh and position
'''
    rng = np.random.default_rng()
    temp = sum(routes,[]) # flatten out routes
    unserved_cust = data.indexes[np.isin(data.indexes,temp, invert = True)]

    if len(unserved_cust) == 0:
        return routes

    cust_insert = rng.choice(unserved_cust, 1 , replace = False)
    insert_veh = rng.integers(len(routes))
    insert_position = rng.integers(1,len(routes[insert_veh])+1) #+1 to allow for empty vehicles
    routes[insert_veh].insert(insert_position, cust_insert[0])
    return routes
'''

def best_insertion_fastest(routes,data,initial_data,parameters):
    '''
    ## Addition of time slacks
    NOTE!!! For the subsequent trips need to consider addition time taken to load fuel delivered.
    '''
    indexes = data.indexes
    # base_obj = objective_function_relaxed(routes, data, initial_data, parameters)
    
    temp = sum(routes,[]) # flatten out routes
    # print('routes',routes)
    # print('temp',temp)
    unserved_cust = list(indexes[np.isin(indexes,temp, invert = True)])
    # print('Unserved:',unserved_cust)
    if len(unserved_cust) == 0:# no unserved customers
        return routes
    prev_k = None
    prev_r = 0
    prev_i = 0
    change_dict= {}
    while len(unserved_cust) > 0:
        # base_obj = objective_function_fast(routes, data, initial_data, parameters)
        routes_trips = []
        for route in routes:
            route= trip_split(route)
            route.append([0,0])
            routes_trips.append(route)
        # print(routes_trips)
       
        # time,time_slack,time_adjustment_list,demand_slack = find_time_slack_new(routes_trips,data,initial_data,parameters)
        '''
        change_dict = {(k,r,i,cust): data.total_demand[cust] - parameters.ZETA*(np.sum(data.distances[[trip[i-1],cust],[cust,trip[i]]])- data.distances[trip[i-1],trip[i]]) 
                     for cust in unserved_cust for k, route in enumerate(routes_trips) 
                     for r, trip in enumerate(route)
                     for i in range(1,len(trip)) }
        change_list = [(k,r,i,cust, data.total_demand[cust] - parameters.ZETA*(np.sum(data.distances[[trip[i-1],cust],[cust,trip[i]]])- data.distances[trip[i-1],trip[i]]) )
                     for cust in unserved_cust for k, route in enumerate(routes_trips) 
                     for r, trip in enumerate(route)
                     for i in range(1,len(trip)) 
                     if data.total_demand[cust] - parameters.ZETA*(np.sum(data.distances[[trip[i-1],cust],[cust,trip[i]]])- data.distances[trip[i-1],trip[i]]) >=0]
        change_list.sort(key=lambda tup: tup[4], reverse=True)'''
        # print(test_dict)
        if prev_k == None:
            for cust in unserved_cust:
                cust_load = data.total_revenue[cust]
                for k, route in enumerate(routes_trips):
                    for r, trip in enumerate(route):
                        for i in range(1,len(trip)):
                            # temp_route = copy.deepcopy(route)
                            # trip.insert(i,cust)
                            obj_diff = cust_load - parameters.ZETA*(data.distances[trip[i-1],cust] + data.distances[cust,trip[i]]- data.distances[trip[i-1],trip[i]])
                            # if obj_diff <0: #only count improving scores
                                # trip.remove(cust)
                                # continue
                            
                            # _ = check_insertion_linear_new(time,time_slack,time_adjustment_list,demand_slack,route,trip,k,r,i,cust,data,initial_data,parameters,log = False)
                            # assert route_check_trips(route, data, initial_data, parameters,k,log= False)  ==  check_insertion_linear_newest(time2,time_slack2,time_adjustment_list2,demand_slack2,route,trip,k,r,i,cust,data,initial_data,parameters,log = False)
                            # if check_insertion_linear_newest(time2,time_slack2,time_adjustment_list2,demand_slack2,route,trip,k,r,i,cust,data,initial_data,parameters):
                            change_dict[(k,r,i,cust)] = obj_diff
                                
                                # change_list_scores.append(obj_diff)
                                # change_list_tuple.append((k,r,i,cust))
                            # trip.remove(cust)
        else:
            for cust in unserved_cust:
                cust_load = data.total_revenue[cust]
                route = routes_trips[prev_k]
                for r in range(prev_r, len(route)): #overwrite entries for subsequent customers
                    trip = route[r]
                    if r == prev_r:
                        for i in range(prev_i,len(trip)):
                            obj_diff = cust_load - parameters.ZETA*(data.distances[trip[i-1],cust] + data.distances[cust,trip[i]]- data.distances[trip[i-1],trip[i]])
                            change_dict[(prev_k,r,i,cust)] = obj_diff
                    else:
                        for i in range(1,len(trip)): 
                            obj_diff = cust_load - parameters.ZETA*(data.distances[trip[i-1],cust] + data.distances[cust,trip[i]]- data.distances[trip[i-1],trip[i]])
                            change_dict[(prev_k,r,i,cust)] = obj_diff

        '''
        if len(change_dict.items()) == 0: #no where to add customers
            
            # unserved_cust.remove(cust)
            # continue
            return routes'''
        # sorted_list = sorted(change_dict.items(), key=lambda item: item[1], reverse = True)
        sorted_list = sorted(change_dict.items(), key=itemgetter(1), reverse = True)
        # print(sorted_list)
        INSERTED = False
        time2,time_slack2,time_adjustment_list2,demand_slack2 = find_time_slack_newest(routes_trips,data,initial_data,parameters)
        for a in sorted_list:
            k,r,i,cust = a[0]
            route = routes_trips[k]
            trip = route[r]

            # trip.insert(i,cust)
            # proper_check = route_check_trips(route, data, initial_data, parameters,k,log= False)
            # trip.remove(cust)
            # assert  proper_check ==  check_insertion_linear_newest(time2,time_slack2,time_adjustment_list2,demand_slack2,route,trip,k,r,i,cust,data,initial_data,parameters,log = False)
            if check_insertion_linear_newest(time2,time_slack2,time_adjustment_list2,demand_slack2,route,trip,k,r,i,cust,data,initial_data,parameters):
                # trip.remove(cust)
                # if a[1] < 0:
                #     print('negative objective diff')
                trip.insert(i,cust)
                prev_k = k
                prev_r = r
                prev_i = i
                # routes_trips[k][r].insert(i,cust)
                unserved_cust.remove(cust)
                # print('Cust inserted', cust)
                INSERTED =True
                for k, route in enumerate(routes_trips): #remove all instances of the inserted customer
                    for r, trip in enumerate(route):
                        for i in range(1,len(trip)):
                            try:
                                del change_dict[(k,r,i,cust)]
                            except: continue
                break
            else: 
                del change_dict[(k,r,i,cust)] #remove customers proven to be infeasible
            # else:
            #     trip.remove(cust)
        if not INSERTED: #no where to add customers
            return routes
        # max_tuple = max(change_dict, key=change_dict.get)
        # max_tuple = change_list_tuple[np.argmax(change_list_scores)]
        # routes_trips[max_tuple[0]][max_tuple[1]].insert(max_tuple[2],max_tuple[3])
        # unserved_cust.remove(max_tuple[3])

        routes = []
        # print("routes-trips",routes_trips)
        for route_trip in routes_trips:
            temp_test = [i for trip in route_trip if len(trip)>2 for i in trip[:-1]]
            while len(temp_test) >0 and temp_test[-1] == 0:
                temp_test.pop(-1)
            if len(temp_test) == 0: #empty route
                temp_test = [0,0]            
                # raise Exception("empty route")
            if temp_test[-1] != 0:
                temp_test.append(0)
            routes.append(temp_test)
    return routes

def random_cust_best_insertion_fastest(routes,data,initial_data,parameters):
    rng = np.random.default_rng()
    indexes = data.indexes
    temp = sum(routes,[]) # flatten out routes
    unserved_cust = list(indexes[np.isin(indexes,temp, invert = True)])
    if len(unserved_cust) == 0:# no unserved customers
        return routes
    while len(unserved_cust) > 0:
        routes_trips = []
        for route in routes:
            route= trip_split(route)
            route.append([0,0])
            routes_trips.append(route)
        # print(routes_trips)
        # time,time_slack,time_adjustment_list,demand_slack = find_time_slack_new(routes_trips,data,initial_data,parameters)
        time2,time_slack2,time_adjustment_list2,demand_slack2 = find_time_slack_newest(routes_trips,data,initial_data,parameters)
        change_dict= {}
        cust = rng.choice(unserved_cust)
        cust_load = data.total_revenue[cust]
        for k, route in enumerate(routes_trips):
            for r, trip in enumerate(route):
                for i in range(1,len(trip)):
                    # trip.insert(i,cust)
                    obj_diff = cust_load - parameters.ZETA*(data.distances[trip[i-1],cust] + data.distances[cust,trip[i]]- data.distances[trip[i-1],trip[i]])
                    # if obj_diff <0: #only count improving scores
                        # trip.remove(cust)
                        # continue
                    # print( route_check_trips(route, data, initial_data, parameters,k))
                    # print(check_insertion_linear_new(time,time_slack,time_adjustment_list,demand_slack,route,trip,k,r,i,cust,data,initial_data,parameters))
                    # _ = check_insertion_linear_new(time,time_slack,time_adjustment_list,demand_slack,route,trip,k,r,i,cust,data,initial_data,parameters,log =False)
                    # assert route_check_trips(route, data, initial_data, parameters,k,log=False) == check_insertion_linear_newest(time2,time_slack2,time_adjustment_list2,demand_slack2,route,trip,k,r,i,cust,data,initial_data,parameters,log =False)
                    # if check_insertion_linear_newest(time2,time_slack2,time_adjustment_list2,demand_slack2,route,trip,k,r,i,cust,data,initial_data,parameters):
 
                    change_dict[(k,r,i,cust)] = obj_diff
                    # trip.remove(cust)
                    # print(trip)
        '''
        if len(change_dict.items()) == 0: #no where to add
            unserved_cust.remove(cust)
            continue'''
        # sorted_list = sorted(change_dict.items(), key=lambda item: item[1], reverse = True)
        sorted_list = sorted(change_dict.items(), key=itemgetter(1), reverse = True)
        # assert sorted_list == sorted_list1
        INSERTED = False
        for a in sorted_list:
            k,r,i,cust = a[0]
            route = routes_trips[k]
            trip = route[r]
            # trip.insert(i,cust)

            # trip.insert(i,cust)
            # proper_check = route_check_trips(route, data, initial_data, parameters,k,log= False)
            # trip.remove(cust)

            # assert proper_check  ==  check_insertion_linear_newest(time2,time_slack2,time_adjustment_list2,demand_slack2,route,trip,k,r,i,cust,data,initial_data,parameters,log = False)
            if check_insertion_linear_newest(time2,time_slack2,time_adjustment_list2,demand_slack2,route,trip,k,r,i,cust,data,initial_data,parameters):
                # trip.remove(cust)
                # routes_trips[k][r].insert(i,cust)
                trip.insert(i,cust)
                unserved_cust.remove(cust)
                # print('Cust inserted', cust)
                INSERTED =True
                break
            # else:
            #     trip.remove(cust)
        if not INSERTED: #no where to add customers
            # unserved_cust.remove(cust)
            return routes
        # max_tuple = max(change_dict, key=change_dict.get)
        # routes_trips[max_tuple[0]][max_tuple[1]].insert(max_tuple[2],max_tuple[3])
        # unserved_cust.remove(max_tuple[3])
        routes = []
        for route_trip in routes_trips:
            temp = [i for trip in route_trip if len(trip)>2 for i in trip[:-1]]
            while len(temp) >0 and temp[-1] == 0:
                temp.pop(-1)
            if len(temp) == 0: #empty route
                temp = [0,0]      
            if temp[-1]!=0:
                temp.append(0)
            '''
            if len(temp) == 0:
                routes.append(temp)
                continue'''
            routes.append(temp)
        # print('Added',routes)
    return routes

def best_regret2_fastest(routes, data, initial_data, parameters):
    indexes = data.indexes
    # base_obj = objective_function(routes, data, initial_data, parameters)
    temp = sum(routes,[]) # flatten out routes
    unserved_cust = list(indexes[np.isin(indexes,temp, invert = True)])
    # print('Unserved:',unserved_cust)
    if len(unserved_cust) == 0:# no unserved customers
        return routes
    # infeasible_locations = defaultdict(list)
    while len(unserved_cust) > 0:
        routes_trips = []
        for route in routes:
            route= trip_split(route)
            route.append([0,0])
            routes_trips.append(route)
        # time,time_slack,time_adjustment_list,demand_slack = find_time_slack_new(routes_trips,data,initial_data,parameters)
        time2,time_slack2,time_adjustment_list2,demand_slack2 = find_time_slack_newest(routes_trips,data,initial_data,parameters)
        change_dict= defaultdict(list)

        for cust in unserved_cust:
            cust_load = data.total_revenue[cust]
            for k, route in enumerate(routes_trips):
                for r, trip in enumerate(route):
                    # infeasible_list = infeasible_locations[(k,r)]
                    # index_list = [i[0] for i in infeasible_list if i[1] == cust]
                    for i in range(1,len(trip)):
                        # trip.insert(i,cust)
                        # if (i,cust) in infeasible_locations[(k,r)]:
                            # continue
                        obj_diff = cust_load - parameters.ZETA*(data.distances[trip[i-1],cust] + data.distances[cust,trip[i]]- data.distances[trip[i-1],trip[i]])
                        # obj_diff = np.sum(data.demands[cust]) - parameters.ZETA*(np.sum(data.distances[[trip[i-1],cust],[cust,trip[i+1]]])- data.distances[trip[i-1],trip[i+1]])
                        # if obj_diff < 0: #only count improving scores
                            # trip.remove(cust)
                            # continue
                        # temp_route = create_trips_single(temp_route, data, initial_data, parameters,k)
                        # _ = check_insertion_linear_new(time,time_slack,time_adjustment_list,demand_slack,route,trip,k,r,i,cust,data,initial_data,parameters)
                        # assert route_check_trips(route, data, initial_data, parameters,k,log=False) == check_insertion_linear_newest(time2,time_slack2,time_adjustment_list2,demand_slack2,route,trip,k,r,i,cust,data,initial_data,parameters,log =False)
                        # if check_insertion_linear_newest(time2,time_slack2,time_adjustment_list2,demand_slack2,route,trip,k,r,i,cust,data,initial_data,parameters):
                        change_dict[cust].append((k,r,i,obj_diff))

                                # change_dict[(k,i,cust)] = new_obj-base_obj
                        # trip.remove(cust)
                    
                    # change_dict[cust].append((k,i,new_obj-base_obj))
        max_regret = 0
        max_tuple = ()
        
        for cust, obj_diff_list in change_dict.items():
            obj_diff_list.sort(key=lambda tup: tup[3], reverse=True)

            max_obj_diff_list = []
            for a in obj_diff_list:

                k,r,i,obj_diff = a
                route = routes_trips[k]
                trip = route[r]
                # trip.insert(i,cust)

                # trip.insert(i,cust)
                # proper_check = route_check_trips(route, data, initial_data, parameters,k,log= False)
                # trip.remove(cust)
                # assert proper_check  ==  check_insertion_linear_newest(time2,time_slack2,time_adjustment_list2,demand_slack2,route,trip,k,r,i,cust,data,initial_data,parameters,log = False)
                
                if check_insertion_linear_newest(time2,time_slack2,time_adjustment_list2,demand_slack2,route,trip,k,r,i,cust,data,initial_data,parameters):
                    # trip.remove(cust)
                    if len(max_obj_diff_list) == 0:
                        max_obj_diff_list.append((k,r,i,cust,obj_diff))
                    elif len(max_obj_diff_list) == 1:#found the top 2 obj diff
                        max_obj_diff_list.append((k,r,i,cust,obj_diff))
                        break
                    else:
                        raise Exception('Error in regret calculation')
                # else:
                    # infeasible_locations[(k,r)].append((i,cust))

            if len(max_obj_diff_list) == 0: # no viable insertion place for customer
                continue
            elif len(max_obj_diff_list) == 1: # only 1 viable insertion
                regret = max_obj_diff_list[0][4]
            elif len(max_obj_diff_list) == 2:
                regret = max_obj_diff_list[0][4] - max_obj_diff_list[1][4]
            # assert regret >= 0
            if regret > max_regret: #check if regret of this cust is the largest
                max_regret = regret
                max_tuple = (max_obj_diff_list[0])

        '''
        for cust, value in change_dict.items():
            value = np.array(value)
            temp = value[:,3]
            max1 = np.max(temp)
            if len(temp) ==1: #if only one valid position, set the other place to 0
                max2 = 0
            else:
                max2 = np.partition(temp,-2)[-2]
            regret = max1-max2
            if regret> max_regret:
                max_regret = regret
                max_index = np.argmax(temp)
                k = int(value[max_index][0])
                r = int(value[max_index][1])
                i = int(value[max_index][2])
                max_tuple = tuple([k,r,i,cust])'''
        # max_tuple = max(change_dict, key=change_dict.get)
        # if change_dict[max_tuple] >0:
            # print(change_dict[max_tuple])
        if len(max_tuple) == 0: #no one to add
            # unserved_cust.remove(cust)
            # continue
            # print("ERROR")
            return routes
        # prev_k,prev_r,prev_i,prev_cust = max_tuple[0],max_tuple[1],max_tuple[2],max_tuple[3]
        # for r in range(prev_r,len(routes_trips[prev_k])): #remove all infeasible checked after inserted
            #  del infeasible_locations[(prev_k,r)]
        routes_trips[max_tuple[0]][max_tuple[1]].insert(max_tuple[2],max_tuple[3])
        # routes[max_tuple[0]].insert(max_tuple[1],max_tuple[2])
        unserved_cust.remove(max_tuple[3])

        routes = []
        for route_trip in routes_trips:
            temp = [i for trip in route_trip if len(trip)>2 for i in trip[:-1]]
            if len(temp) ==0: #empty route
                temp = [0,0]
            elif temp[-1]!=0:
                temp.append(0)
            routes.append(temp)
    return routes

def best_regret2(routes, data, initial_data, parameters):
    indexes = data.indexes
    # base_obj = objective_function(routes, data, initial_data, parameters)
    temp = sum(routes,[]) # flatten out routes
    unserved_cust = list(indexes[np.isin(indexes,temp, invert = True)])
    # print('Unserved:',unserved_cust)
    if len(unserved_cust) == 0:# no unserved customers
        return routes
    # infeasible_locations = defaultdict(list)
    prev_k = None
    prev_r = 0
    prev_i = 0
    change_dict= {}
    while len(unserved_cust) > 0:
        routes_trips = []
        for route in routes:
            route= trip_split(route)
            route.append([0,0])
            routes_trips.append(route)
        # time,time_slack,time_adjustment_list,demand_slack = find_time_slack_new(routes_trips,data,initial_data,parameters)
       
        
        if prev_k == None:
            for cust in unserved_cust:
                cust_load = data.total_revenue[cust]
                for k, route in enumerate(routes_trips):
                    for r, trip in enumerate(route):
                        for i in range(1,len(trip)):
                            obj_diff = cust_load - parameters.ZETA*(data.distances[trip[i-1],cust] + data.distances[cust,trip[i]]- data.distances[trip[i-1],trip[i]])
                            change_dict[(k,r,i,cust)] = obj_diff

        else:
            for cust in unserved_cust:
                cust_load = data.total_revenue[cust]
                route = routes_trips[prev_k]
                for r in range(prev_r, len(route)): #overwrite entries for subsequent customers
                    trip = route[r]
                    if r == prev_r:
                        for i in range(prev_i,len(trip)):
                            obj_diff = cust_load - parameters.ZETA*(data.distances[trip[i-1],cust] + data.distances[cust,trip[i]]- data.distances[trip[i-1],trip[i]])
                            change_dict[(prev_k,r,i,cust)] = obj_diff
                    else:
                        for i in range(1,len(trip)): 
                            obj_diff = cust_load - parameters.ZETA*(data.distances[trip[i-1],cust] + data.distances[cust,trip[i]]- data.distances[trip[i-1],trip[i]])
                            change_dict[(prev_k,r,i,cust)] = obj_diff

        max_regret = 0
        max_tuple = ()
        sorted_list = sorted(change_dict.items(), key=itemgetter(1), reverse = True)
        sorted_dict = defaultdict(list)
        [sorted_dict[a[0][3]].append(a) for a in sorted_list]
        time2,time_slack2,time_adjustment_list2,demand_slack2 = find_time_slack_newest(routes_trips,data,initial_data,parameters)
        
        for unserved_customer in sorted_dict.values(): #check for each unserved
            max_obj_diff_list = []
            for a in unserved_customer:
                k,r,i,cust = a[0]
                obj_diff = a[1]
                route = routes_trips[k]
                trip = route[r]
                if check_insertion_linear_newest(time2,time_slack2,time_adjustment_list2,demand_slack2,route,trip,k,r,i,cust,data,initial_data,parameters):
                    if len(max_obj_diff_list) == 0:
                        max_obj_diff_list.append((k,r,i,cust,obj_diff))
                    elif len(max_obj_diff_list) == 1:#found the top 2 obj diff
                        max_obj_diff_list.append((k,r,i,cust,obj_diff))
                        break
                    else:
                        raise Exception('Error in regret calculation')
                else:
                    del change_dict[(k,r,i,cust)] #remove customers proven to be infeasible

            if len(max_obj_diff_list) == 0: # no viable insertion place for customer
                continue
            elif len(max_obj_diff_list) == 1: # only 1 viable insertion
                regret = max_obj_diff_list[0][4]
            elif len(max_obj_diff_list) == 2:
                regret = max_obj_diff_list[0][4] - max_obj_diff_list[1][4]
            # assert regret >= 0
            if regret > max_regret: #check if regret of this cust is the largest
                max_regret = regret
                max_tuple = (max_obj_diff_list[0])

        if len(max_tuple) == 0: #no one to add
            # unserved_cust.remove(cust)
            # continue
            # print("ERROR")
            return routes
        prev_k,prev_r,prev_i,prev_cust = max_tuple[0],max_tuple[1],max_tuple[2],max_tuple[3]
        routes_trips[max_tuple[0]][max_tuple[1]].insert(max_tuple[2],max_tuple[3])

        unserved_cust.remove(max_tuple[3])
        for k, route in enumerate(routes_trips): #remove all instances of the inserted customer
                for r, trip in enumerate(route):
                    for i in range(1,len(trip)):
                        try:
                            del change_dict[(k,r,i,prev_cust)]
                        except: continue

        routes = []
        for route_trip in routes_trips:
            temp = [i for trip in route_trip if len(trip)>2 for i in trip[:-1]]
            if len(temp) ==0: #empty route
                temp = [0,0]
            elif temp[-1]!=0:
                temp.append(0)
            routes.append(temp)
    return routes