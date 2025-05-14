import numpy as np
from model_check import trip_split, find_time_slack, check_insertion_constant
from collections import defaultdict
import logging
from operator import itemgetter

def best_insertion(routes,data,initial_data,parameters):
    indexes = data.indexes
    temp = sum(routes,[]) # flatten out routes
    unserved_cust = list(indexes[np.isin(indexes,temp, invert = True)])
    if len(unserved_cust) == 0:# no unserved customers
        return routes
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

        sorted_list = sorted(change_dict.items(), key=itemgetter(1), reverse = True)
        INSERTED = False
        time2,time_slack2,time_adjustment_list2,demand_slack2 = find_time_slack(routes_trips,data,initial_data,parameters)
        for a in sorted_list:
            k,r,i,cust = a[0]
            route = routes_trips[k]
            trip = route[r]

            if check_insertion_constant(time2,time_slack2,time_adjustment_list2,demand_slack2,route,trip,k,r,i,cust,data,initial_data,parameters):
                trip.insert(i,cust)
                prev_k = k
                prev_r = r
                prev_i = i
                unserved_cust.remove(cust)
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

        if not INSERTED: #no where to add customers
            return routes
        
        routes = []
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

def random_cust_best_insertion(routes,data,initial_data,parameters):
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
        time2,time_slack2,time_adjustment_list2,demand_slack2 = find_time_slack(routes_trips,data,initial_data,parameters)
        change_dict= {}
        cust = rng.choice(unserved_cust)
        cust_load = data.total_revenue[cust]
        for k, route in enumerate(routes_trips):
            for r, trip in enumerate(route):
                for i in range(1,len(trip)):
                    obj_diff = cust_load - parameters.ZETA*(data.distances[trip[i-1],cust] + data.distances[cust,trip[i]]- data.distances[trip[i-1],trip[i]])
                    change_dict[(k,r,i,cust)] = obj_diff

        sorted_list = sorted(change_dict.items(), key=itemgetter(1), reverse = True)
        INSERTED = False
        for a in sorted_list:
            k,r,i,cust = a[0]
            route = routes_trips[k]
            trip = route[r]
            if check_insertion_constant(time2,time_slack2,time_adjustment_list2,demand_slack2,route,trip,k,r,i,cust,data,initial_data,parameters):
                trip.insert(i,cust)
                unserved_cust.remove(cust)
                INSERTED =True
                break

        if not INSERTED: #no where to add customers
            return routes
       
        routes = []
        for route_trip in routes_trips:
            temp = [i for trip in route_trip if len(trip)>2 for i in trip[:-1]]
            while len(temp) >0 and temp[-1] == 0:
                temp.pop(-1)
            if len(temp) == 0: #empty route
                temp = [0,0]      
            if temp[-1]!=0:
                temp.append(0)
            routes.append(temp)
    return routes

def best_regret2(routes, data, initial_data, parameters):
    indexes = data.indexes
    temp = sum(routes,[]) # flatten out routes
    unserved_cust = list(indexes[np.isin(indexes,temp, invert = True)])
    if len(unserved_cust) == 0:# no unserved customers
        return routes
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
        time2,time_slack2,time_adjustment_list2,demand_slack2 = find_time_slack(routes_trips,data,initial_data,parameters)
        
        for unserved_customer in sorted_dict.values(): #check for each unserved
            max_obj_diff_list = []
            for a in unserved_customer:
                k,r,i,cust = a[0]
                obj_diff = a[1]
                route = routes_trips[k]
                trip = route[r]
                if check_insertion_constant(time2,time_slack2,time_adjustment_list2,demand_slack2,route,trip,k,r,i,cust,data,initial_data,parameters):
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

            if regret > max_regret: #check if regret of this cust is the largest
                max_regret = regret
                max_tuple = (max_obj_diff_list[0])

        if len(max_tuple) == 0: #no one to add
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