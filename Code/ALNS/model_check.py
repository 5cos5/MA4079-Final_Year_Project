import numpy as np
import logging

def cust_no_index(cust_no:int, data):
    '''
    converts customer_number to index in data
    '''
    temp = np.where(data.cust_no == cust_no)
    #print (len(temp[0]))
    if len(temp[0]) ==1:
        return temp[0][0]
    else:
        logging.warning('Node not in customer list')
        return -1

def trip_split(route):
    trip =[]
    try:
        temp = [route[0]]
    except:
        temp = []
    for i in range(1, len(route)):
        temp.append(route[i])
        if route[i] == 0:
            trip.append(temp)
            temp = [0]
    return trip

def replace_consecutive_depots(routes):
    for k, route_veh in enumerate(routes):
        if len(route_veh) >2:
            to_remove = []
            for i, cust in enumerate(route_veh[:-1]):
                next_cust = route_veh[i+1]
                if cust == next_cust and (cust == 1 or cust==0):
                    to_remove.append(i)
            for i in to_remove[::-1]:
                try:
                # print('fixed')
                    del route_veh[i]
                except:
                    print('Routes', routes)
                    print('to remove', to_remove)
                    print('cust to remove', i)
                    raise Exception('Cannot replace conseceutive depots')
    return routes



def find_time_slack(routes_trips,data,initial_data,parameters,log =False):
    '''
    Takes in routes in the form of
    [[[0,x,x,0],[0,x,x,x,0],[0,0]],
    [[0,x,x,0],[0,x,x,x,0],[0,0]],
    [[0,x,x,0],[0,x,x,x,0],[0,0]]]

    finds the time of each visit, waiting time at each visit (time_adjustment) and time slack (time-end_time_window) for each trip
    subsequently, it calculates the min time slack at that insertion position (time_slacks_fixed). 
    This can then be used to compare with time slack needed for an insertion in the linear check.
    
    '''
    time_slacks_list = []
    time_adjustment_list = []
    time_list = []
    loading_slack_list = []
    initial_route = [cust_no_index(cust,data) for cust in initial_data.initial_route]

    for k, route in enumerate(routes_trips):
        time_slacks_veh = []
        time_adjustment_veh =[]
        total_waiting_time = 0
        loading_slacks_veh = []
        time_veh = []
        time = initial_data.initial_time[k]
        load = np.array(initial_data.initial_loading[k])
        MAX_CAPACITY = tuple(parameters.MAX_CAPACITY)
        for r,trip in enumerate(route):
            time_slacks_trip = []
            time_adjustment_trip =[]
            time_trip = []
            if r == 0: #first trip
                node = initial_route[k]
                time_adjustment = 0
                if time < data.start_time_windows[node]:
                        time_adjustment = data.start_time_windows[node] - time
                        total_waiting_time += time_adjustment
                        time = data.start_time_windows[node]
                
                time_trip.append(time)
                time_slacks_trip.append(data.end_time_windows[node] - time)
                time_adjustment_trip.append(time_adjustment)
                loading_slacks_trip = np.array(initial_data.initial_loading[k]) - np.sum(data.demands[trip], axis=0)
                if (loading_slacks_trip <parameters.EMPTY_CAPACITY).any():
                    raise Exception ("Trip is over load")
                
                prev_node = node
            else: # not the first trip, perform loading
                prev_node = 0
                total_to_load = np.array(MAX_CAPACITY)-np.array(load) #find amount of fuel to load\
                loading_time = parameters.DEPOT_LOADING_SCALAR*np.sum(total_to_load)
                time += loading_time #add loading time
                load = MAX_CAPACITY #set capacity back to maximum
                time_trip.append(time)
                time_adjustment_trip.append(0)
                time_slacks_trip.append(data.end_time_windows[prev_node] - time)
                loading_slacks_trip = MAX_CAPACITY - np.sum(data.demands[trip], axis=0)

            for i, node in enumerate(trip[1:]):
                if node == 0: # back at the depot (end of trip)
                    travel_time = data.distances[prev_node,0] #travel back to depot
                    time += travel_time
                    prev_node = 0
                    time_trip.append(time)
                    time_adjustment_trip.append(0)
                    time_slacks_trip.append(data.end_time_windows[prev_node] - time)
                else: #travelling to next node
                    travel_time = data.distances[prev_node,node]
                    time+= travel_time
                    time_adjustment = 0
                    
                    if time < data.start_time_windows[node]:
                        time_adjustment = data.start_time_windows[node] - time
                        total_waiting_time += time_adjustment
                        time = data.start_time_windows[node]

                    time_slacks_trip.append(data.end_time_windows[node] - time)
                    time_trip.append(time)
                    time_adjustment_trip.append(time_adjustment)
                    service_time = data.service_time[node]
                    time += service_time
                    if time > data.end_time_windows[node]:
                        if log:
                            logging.error("In final route reach customer after end time window")
                    service_load = np.array(data.demands[node])
                    load -= service_load
                prev_node = node
            time_veh.append(time_trip)
            time_slacks_veh.append(time_slacks_trip)
            time_adjustment_veh.append(time_adjustment_trip)
            loading_slacks_veh.append(loading_slacks_trip)
        time_slacks_list.append(time_slacks_veh)
        time_adjustment_list.append(time_adjustment_veh)
        time_list.append(time_veh)
        loading_slack_list.append(loading_slacks_veh)
    time_slacks_fixed = []
    
    for k, time_slacks_veh in enumerate(time_slacks_list):
        time_slacks_veh_fixed = []
        newer_flat_time_adustment_list = [item for row in time_adjustment_list[k] for item in row]
        newer_flat_time_slacks_list = [item for row in time_slacks_list[k] for item in row]
        time_adj = time_adjustment_list[k]
        temp_all_prev_trips_list_newer = np.array([x+sum(newer_flat_time_adustment_list[:y+1]) for y, x in enumerate(newer_flat_time_slacks_list)])
        
        count_flat = len(time_slacks_veh[0])
        for r, time_slacks_trip in enumerate(time_slacks_veh):
            trip_length = len(time_slacks_trip)
            time_slacks_trip_fixed = []
           
            if r < len(time_slacks_veh) -2: #ignore the last trip and dummy last
                temp_all_prev_trips_list_newer_temp6 = temp_all_prev_trips_list_newer[trip_length:-2] - sum(time_adjustment_list[k][r])
                temp6 = min(temp_all_prev_trips_list_newer_temp6)

            else: # this is not important as it wont be checked
                temp6 = -1
            
          
            for i, time_slack in enumerate(time_slacks_trip):
                time_slacks_node_fixed = []
                temp3 = min(temp_all_prev_trips_list_newer[i:trip_length])
                if r < len(time_slacks_veh)-2: #ignore the last trip and dummy last
            
                    temp5 = min(temp_all_prev_trips_list_newer[trip_length:-2])

                else: # this is not impt as it wont be checked
                    temp5 = -1


                
                temp_all_prev_trips_list_newer -= time_adjustment_list[k][r][i]
                time_slacks_trip_fixed.append((temp3,temp5,temp6))
            temp_all_prev_trips_list_newer = temp_all_prev_trips_list_newer[trip_length:]

            if r < len(time_slacks_veh)-2:
                count_flat += len(time_slacks_veh[r+1])

            time_slacks_veh_fixed.append(time_slacks_trip_fixed)
        time_slacks_fixed.append(time_slacks_veh_fixed)

    return time_list, time_slacks_fixed,time_adjustment_list,loading_slack_list

def check_insertion_linear(time,time_slack,time_adjustment,demand_slack,route,trip,k,r,i,cust,data,initial_data,parameters,log = False, THRESHOLD = 0.000001):
    if (demand_slack[k][r]-data.demands[cust]<parameters.EMPTY_CAPACITY).any():
        # if log:    
            # logging.warning("Over Demand")
        return False

    time_visit = time[k][r][i-1]
    time_visit += data.service_time[trip[i-1]]
    time_visit += data.distances[trip[i-1],cust]
    time_slack_to_add = 0
    if time_visit < data.start_time_windows[cust]:
        time_slack_to_add = data.start_time_windows[cust] - time_visit

    if time_visit - data.end_time_windows[cust] > THRESHOLD:
        # if log:
            # logging.warning("Newest Time past time windows")
        return False
    
    time_slack_needed_current_trip = data.service_time[cust] + data.distances[trip[i-1],cust] + data.distances[cust,trip[i]] - data.distances[trip[i-1],trip[i]] + time_slack_to_add
    time_slack_avail_current_trip = time_slack[k][r][i]

    if time_slack_needed_current_trip - time_slack_avail_current_trip[0] > THRESHOLD :
        # if log:
            # logging.warning('Newest Not enough time slack for current trip')
        return False

    if r < len(route)-2: # if the trip we are analysing is not the last trip b4 dummy
        loading_time = parameters.DEPOT_LOADING_SCALAR*np.sum(data.demands[cust])
        time_slack_needed_next_trip = time_slack_needed_current_trip + loading_time
        if time_slack_needed_next_trip - time_slack_avail_current_trip[1] > THRESHOLD:
            # if log:
                # logging.warning('Newest Not enough time slack for next trip')
            return False
        
        if loading_time - time_slack_avail_current_trip[2] > THRESHOLD:
            # if log:
                # logging.warning('Newest Not enough time slack for next trip due to loading')
            return False
    return True

def route_check_single(route, data, initial_data, parameters,k,log=False):
    '''
    ### Input: Route from a single vehicle in [0,x,x,0,x,x,0] form (not split into trips yet)
'''
    route_validity = True
    routes_trip = trip_split(route) #split into trips
    
    return route_check_trips(routes_trip, data, initial_data,parameters,k, log= log)

def route_check_trips(route, data, initial_data, parameters,k,log=False):
    '''
    ### Input: route of a single vehicle, already split into the different trips
    ### In the form [[0,x,x,0],[0,x,x,x,0]]
    '''
    route_validity = True

    # for k, route_veh in enumerate(routes):
    # route_veh = [cust_no_index(cust,data) for cust in route_veh]
    initial_route = [cust_no_index(cust,data) for cust in initial_data.initial_route[k]]
    # print(initial_route)
    # print('Route_veh:',route_veh)
    for z, initial_node in enumerate(initial_route): # check if route follows inital route
        if initial_node != route[0][z]:
            route_validity =False
            if log:
                logging.error("route does not follow inital route")
            return route_validity

    # routes_trip = trip_split(route) #split into trips
    routes_trip = route
    for r, route_trip in enumerate(routes_trip):
        if not (route_trip[-1] == 0): #route end in depot
            route_validity = False
            if log:
                logging.error("route does not end in depot")
            return route_validity

    # print('routes_trip', routes_trip)

        ### CHECKING START AND END TIME WINDOWS
    time = initial_data.initial_time[k]
    load = np.array(initial_data.initial_loading[k])
    MAX_CAPACITY = tuple(parameters.MAX_CAPACITY)
    for r, route_trip in enumerate(routes_trip):

        # print ('trip:',route_trip)
        prev_node = 0
        if r != 0: #not the first trip, perfomr loading
            # print(MAX_CAPACITY)
            total_to_load = np.array(MAX_CAPACITY)-np.array(load) #find amount of fuel to load
            loading_time = parameters.DEPOT_LOADING_SCALAR*sum(total_to_load)
            time += loading_time #add loading time
            # print(time)
            load = np.array(MAX_CAPACITY) #set capacity back to maximum

        if len(route_trip) >2: #ignore empty trips
            for a, node in enumerate(route_trip[1:]):
                if node == 0: # node is back at the depot
                    travel_time = data.distances[prev_node,0] #travel back to depot
 
                    time += travel_time
                    # print ('Last depot time', time)
                    #ADDED THIS
                    if round(time,5) > data.end_time_windows[node]:
                        route_validity =False
                        if log:
                            logging.error("route reach last depot after end time window")
                        return route_validity
                    # print (time)
                else:
                    #travelling to next location
                    travel_time = data.distances[prev_node,node]
                    time+= travel_time
                    #if reach before start time window, set time to start time window
                    if time < data.start_time_windows[node]:
                        # print (data.start_time_windows[node])
                        time = data.start_time_windows[node]

                    # print('Node', node)
                    # print('time b4',time)
                    # print ('end_time_window',data.end_time_windows[node] )

                    if round(time,5) > data.end_time_windows[node]:
                        route_validity =False
                        if log:
                            logging.error("route reach customer after end time window")
                        return route_validity
                    #calcualte service time and loading

                    service_time = data.service_time[node]
                    time+=service_time
                    
                    
                    # print('time af', time)
                    
                    #calculate loading
                    service_load = np.array(data.demands[node])
                    # print("Cuirrent load", load)
                    # print("service load", service_load)
                    
                    if (service_load>MAX_CAPACITY).any():
                        route_validity =False
                        if log:
                            logging.error("Load at node exceeds vehicle capacity")
                        return route_validity
                    load -= service_load
                    if (load<parameters.EMPTY_CAPACITY).any():
                        route_validity =False
                        if log:
                            logging.error("Vehicle does not have enough capacity")
                        return route_validity
                    prev_node = node
    return route_validity

def route_check(routes, data, initial_data, parameters,log=False, length_check = True):
    route_validity = True

    if (len(routes) != len(initial_data.initial_route)) and length_check:
        logging.error("Length of routes and intial data is not the same")
        raise Exception("Length of routes and intial data is not the same")

    for k, route_veh in enumerate(routes):
        # route_veh = [cust_no_index(cust,data) for cust in route_veh]
        initial_route = [cust_no_index(cust,data) for cust in initial_data.initial_route[k]]
        # print(initial_route)
        # print('Route_veh:',route_veh)
        for z, initial_node in enumerate(initial_route): # check if route follows inital route
            if initial_node != route_veh[z]:
                route_validity =False
                if log:
                    logging.error("route does not follow inital route")
                return route_validity

        routes_trip = trip_split(route_veh) #split into trips
        
        for r, route_trip in enumerate(routes_trip):
            if not (route_trip[-1] == 0): #route  end in depot
                route_validity = False
                if log:
                    logging.error("route does not  end in depot")
                return route_validity



            ### CHECKING START AND END TIME WINDOWS
        time = initial_data.initial_time[k]
        load = np.array(initial_data.initial_loading[k])
        MAX_CAPACITY = tuple(parameters.MAX_CAPACITY)
        for r, route_trip in enumerate(routes_trip):
            # print ('trip:',r)
            prev_node = 0
            if r != 0: #not the first trip, perfomr loading
                # print(MAX_CAPACITY)
                total_to_load = np.array(MAX_CAPACITY)-np.array(load) #find amount of fuel to load
                loading_time = parameters.DEPOT_LOADING_SCALAR*sum(total_to_load)
                time += loading_time #add loading time
                # print(time)
                load = np.array(MAX_CAPACITY) #set capacity back to maximum
                
            for a, node in enumerate(route_trip[1:]):
                if node == 0: # node is back at the depot
                    travel_time = data.distances[prev_node,0] #travel back to depot
                    time += travel_time
                    # print (time)

                    #ADDED THIS
                    if round(time,5) > data.end_time_windows[node]:
                        route_validity =False
                        if log:
                            logging.error("route reach last depot after end time window")
                        return route_validity
                
                else:
                    #travelling to next location
                    travel_time = data.distances[prev_node,node]
                    time+= travel_time
                    #if reach before start time window, set time to start time window
                    if time < data.start_time_windows[node]:
                        time = data.start_time_windows[node]
                    if round(time,5) > data.end_time_windows[node]:
                        route_validity =False
                        if log:
                            print('Location',k,r,node)
                            print('Time', time)
                            print('End time window', data.end_time_windows[node])
                            logging.error("route reach customer after end time window")
                        return route_validity
                    # print(time)
                    #calcualte service time and loading
                    service_time = data.service_time[node]
                    time+=service_time
                    
                    #calculate loading
                    service_load = np.array(data.demands[node])
                    if (service_load>MAX_CAPACITY).any():
                        route_validity =False
                        if log:
                            logging.error("Load at node exceeds vehicle capacity")
                        return route_validity
                    load -= service_load
                    if (load<parameters.EMPTY_CAPACITY).any():
                        route_validity =False
                        if log:
                            logging.error("Vehicle does not have enough capacity")
                        return route_validity
                    prev_node = node
    return route_validity

def objective_function_fast(routes, data, initial_data, parameters):
    '''
    ### This works. Can use as a drop in replacement of objective_function
    '''
    total_distance_travelled =0
    distances = data.distances
    routes_flat = [i for route in routes for i in route]
    for route_veh in routes:
        total_distance_travelled += np.sum(distances[route_veh[:-1],route_veh[1:]])
    # return total_distance_travelled
    # print('old', np.sum(data.demands[routes_flat])-parameters.ZETA*total_distance_travelled)
    # print('new', data.total_revenue[routes_flat] -parameters.ZETA*total_distance_travelled)
    return np.sum(data.demands[routes_flat])-parameters.ZETA*total_distance_travelled

def objective_function(routes, data, initial_data, parameters):
    total_demand_delivered = [0] * parameters.DEMAND_TYPES
    total_cust_visited = 0
    total_distance_travelled = 0
    for k, route_veh in enumerate(routes):
        routes_trip = trip_split(route_veh) #split into trips
        for r, route_trip in enumerate(routes_trip):
            prev_node = route_trip[0]
            total_demand_delivered += data.demands[prev_node]
            if sum(data.demands[prev_node]) >0:
                total_cust_visited+= 1
            for a, node in enumerate(route_trip[1:]):
                if node ==0:
                    total_distance_travelled += data.distances[prev_node,0]
                else:
                    total_cust_visited+= 1
                    total_demand_delivered += data.demands[node]
                    total_distance_travelled += data.distances[prev_node,node]
                    prev_node = node
    return sum(total_demand_delivered*parameters.PRICE)-parameters.ZETA*total_distance_travelled

def distance_travelled(routes, data, initial_data, parameters):
    '''
    ### To find distance travelled
    '''
    total_distance_travelled =0
    distances = data.distances
    routes_flat = [i for route in routes for i in route]
    for route_veh in routes:
        # print(route_veh)
        # print(np.sum(distances[route_veh[:-1],route_veh[1:]]))
        total_distance_travelled += np.sum(distances[route_veh[:-1],route_veh[1:]])
    return total_distance_travelled
    # return np.sum(data.demands[routes_flat])

def goods_delivered(routes, data, initial_data, parameters):
    '''
    ### To find distance travelled
    '''
    total_distance_travelled =0
    distances = data.distances
    routes_flat = [i for route in routes for i in route]
    '''
    for route_veh in routes:
        # print(route_veh)
        # print(np.sum(distances[route_veh[:-1],route_veh[1:]]))
        total_distance_travelled += np.sum(distances[route_veh[:-1],route_veh[1:]])'''
    # return total_distance_travelled
    return np.sum(data.demands[routes_flat])

