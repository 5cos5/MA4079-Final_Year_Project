from read_data_expt2 import Input, InitialData, InputParameters
import numpy as np
import logging
from decimal import Decimal

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
                if cust == next_cust  and (cust == 1 or cust==0):
                    to_remove.append(i)
            for i in to_remove[::-1]:
                # print('fixed')
                try:
                    del route_veh[i]
                except:
                    print('Routes', routes)
                    print('to remove', to_remove)
                    print('cust to remove', i)
                    raise Exception('Cannot replace conseceutive depots')
    return routes

def find_time_slack_new(routes_trips,data,initial_data,parameters,log =False):
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
        # incorrect_time = initial_data.initial_time[k]
        load = np.array(initial_data.initial_loading[k])
        MAX_CAPACITY = tuple(parameters.MAX_CAPACITY)
        total_service_time = 0
        for r,trip in enumerate(route):
            time_slacks_trip = []
            time_adjustment_trip =[]
            time_trip = []
            if r == 0: #first trip
                node = initial_route[k]
                # node = cust_no_index(initial_data.initial_route[k][0],data)
                time_adjustment = 0

                if time < data.start_time_windows[node]:
                        time_adjustment = data.start_time_windows[node] - time
                        total_waiting_time += time_adjustment
                        time = data.start_time_windows[node]
                
                loading_time = parameters.DEPOT_LOADING_SCALAR*np.sum(data.service_time[trip])
                time += loading_time
                time_trip.append(time)
                time_slacks_trip.append(data.end_time_windows[node] - time)
                time_adjustment_trip.append(time_adjustment)
                loading_slacks_trip = np.array(initial_data.initial_loading[k]) - np.sum(data.demands[trip], axis=0)
                if (loading_slacks_trip <parameters.EMPTY_CAPACITY).any():
                    raise Exception ("Trip is over load")
                
                prev_node = node
            else: # not the first trip, perform loading
                prev_node = 0
                # total_to_load = np.array(MAX_CAPACITY)-np.array(load) #find amount of fuel to load\
                loading_time = parameters.DEPOT_LOADING_SCALAR*np.sum(data.service_time[trip])
                total_service_time = 0
                # loading_time = parameters.DEPOT_LOADING_SCALAR*np.sum(total_to_load)
                time += loading_time #add loading time
                # incorrect_time += loading_time
                load = MAX_CAPACITY #set capacity back to maximum
                # print("Time",time)
                time_trip.append(time)
                time_adjustment_trip.append(0)
                time_slacks_trip.append(data.end_time_windows[prev_node] - time)
                loading_slacks_trip = MAX_CAPACITY - np.sum(data.demands[trip], axis=0)

            for i, node in enumerate(trip[1:]):
                if node == 0: # back at the depot (end of trip)
                    travel_time = data.distances[prev_node,0] #travel back to depot
                    time += travel_time
                    # incorrect_time += travel_time
                    prev_node = 0
                    time_trip.append(time)
                    time_adjustment_trip.append(0)
                    time_slacks_trip.append(data.end_time_windows[prev_node] - time)
                else: #travelling to next node
                    travel_time = data.distances[prev_node,node]
                    time+= travel_time
                    # incorrect_time += travel_time
                    time_adjustment = 0
                    
                    if time < data.start_time_windows[node]:
                        time_adjustment = data.start_time_windows[node] - time
                        total_waiting_time += time_adjustment
                        time = data.start_time_windows[node]

                    time_slacks_trip.append(data.end_time_windows[node] - time)
                    time_trip.append(time)
                    # time_adjustment_trip.append(0)
                    time_adjustment_trip.append(time_adjustment)
                    #
                    # time_slacks_trip.append(data.end_time_windows[node] - time)
                    # print('time_trip',time_trip)
                    service_time = data.service_time[node]
                    time += service_time
                    total_service_time += service_time
                    # incorrect_time += service_time
                    if time > data.end_time_windows[node]:
                        if log:
                            logging.error("In final route reach customer after end time window")
                    service_load = np.array(data.demands[node])
                    load -= service_load
                prev_node = node
            # print(time_trip)
            time_veh.append(time_trip)
            time_slacks_veh.append(time_slacks_trip)
            time_adjustment_veh.append(time_adjustment_trip)
            loading_slacks_veh.append(loading_slacks_trip)
        time_slacks_list.append(time_slacks_veh)
        time_adjustment_list.append(time_adjustment_veh)
        time_list.append(time_veh)
        loading_slack_list.append(loading_slacks_veh)
    # print('Preliminary', time_slacks_list)

    time_slacks_fixed = []
    
    for k, time_slacks_veh in enumerate(time_slacks_list):
        time_slacks_veh_fixed = []
        # flat_time_adustment_list = [item for row in time_adjustment_list[k] for item in row]
        # print('flat_time_adustment_list',flat_time_adustment_list)
        flat_time_adustment_list = [item for row in time_adjustment_list[k] for item in row]
        flat_time_slacks_list = [item for row in time_slacks_list[k] for item in row]
        temp_all_prev_trips_list = np.array([x+sum(flat_time_adustment_list[:y+1]) for y, x in enumerate(flat_time_slacks_list)])
        
        for r, time_slacks_trip in enumerate(time_slacks_veh):
            trip_length = len(time_slacks_trip)
            time_slacks_trip_fixed = []
            time_adj_route = 0
            temp_depot_loading = np.array(time_slacks_trip)
            for i, time_slack in enumerate(time_slacks_trip):
                # if i != 0:
                    # time_slacks_node_fixed = []
                    # print(i)
                    # print('Time adjustment',time_adjustment_list[k][r])
                    # print('Time slacks',time_slacks_veh[r])
                    # print('Time', time_list[k][r])

                    # temp_only_current_trip = [x+sum(time_adjustment_list[k][r][i:i+y+1]) for y, x in enumerate(time_slacks_veh[r][i:])]
                    # # print('current_trip',temp_only_current_trip )
                    # temp_all_prev_trips = [x+sum(time_adjustment_list[k][r][:y]) for y, x in enumerate(time_slacks_veh[r][:i])]
                    # temp1 = round(min(temp_all_prev_trips),5) #time slack avail from previous nodes (for extra delivery time at depot)
                    # temp2 = round(min(temp_only_current_trip),5) #time slack avail from current node onwards
                    # time_slacks_node_fixed.append((temp1, temp2))
                if i!=0:
                    temp_depot_loading[i-1:] += time_adjustment_list[k][r][i-1]
                    temp3 = min(temp_depot_loading[:i])
                else: #can ignore this case i believe
                    temp3 = temp_depot_loading[i]

                temp4 = min(temp_all_prev_trips_list[i:])
                temp_all_prev_trips_list -= time_adjustment_list[k][r][i]
                
                time_slacks_trip_fixed.append((temp3,temp4))

            temp_all_prev_trips_list = temp_all_prev_trips_list[trip_length:]


            time_slacks_veh_fixed.append(time_slacks_trip_fixed)
        time_slacks_fixed.append(time_slacks_veh_fixed)
    # print('Fixed Time Slack', time_slacks_fixed)
    # raise Exception

    return time_list, time_slacks_fixed,time_adjustment_list,loading_slack_list
                
def check_insertion_linear_new(time,time_slack,time_adjustment,demand_slack,route,trip,k,r,i,cust,data,initial_data,parameters,log = False, THRESHOLD = 0.000001):
    # print(demand_slack[k][r])
    if (demand_slack[k][r]-data.demands[cust]<parameters.EMPTY_CAPACITY).any():
        # print('demand slack', demand_slack[k][r])
        if log:    
            logging.warning("Over Demand")
        return False

    time_visit = time[k][r][i-1]
    # print('rouite', route)
    # print('trip', trip)
    # print('time',time)
    # print('inserted_cust', cust)
    # print('B4 time',time_visit)
    # print('Time Slack', time_slack[k][r][i][0])
    if parameters.DEPOT_LOADING_SCALAR * data.service_time[cust] -  time_slack[k][r][i][0] > THRESHOLD:
        if log:
            logging.warning('Not enough time slack for loading')
        return False
    
    time_due_to_depot_loading = parameters.DEPOT_LOADING_SCALAR*data.service_time[cust] - sum(time_adjustment[k][r][:i])
    if time_due_to_depot_loading >0:
        time_visit += time_due_to_depot_loading
    else:
        time_due_to_depot_loading = 0

    time_visit += data.service_time[trip[i-1]]
    time_visit += data.distances[trip[i-1],cust]
    # print('Af time', time_visit)
    time_slack_to_add = 0
    # print(time_visit)
    if time_visit < data.start_time_windows[cust]:
        # print("START_TIME_WINDOWS", data.start_time_windows[cust])
        time_slack_to_add = data.start_time_windows[cust] - time_visit
        time_visit = data.start_time_windows[cust]
        # print('TIME SLACK ADDED',time_slack_to_add )
    # print("Visit time",time_visit)
    # print("End time window",data.end_time_windows[cust])
    if time_visit -data.end_time_windows[cust] > THRESHOLD :
        # print(time)
        # print(trip)
        # print("i",i)
        # print("Visit time",time_visit)
        # print("End time window",data.end_time_windows[cust])
        if log:
            logging.warning("Time past time windows")
        return False
    # print('AF time', time_visit)
    # print('time_due_to_depot_loading',time_due_to_depot_loading)
    # print('service', data.service_time[cust])
    # print('distance', data.distances[trip[i-1],cust] + data.distances[cust,trip[i+1]] - data.distances[trip[i-1],trip[i+1]] )
    time_slack_needed_current_trip = time_due_to_depot_loading + data.service_time[cust] + data.distances[trip[i-1],cust] + data.distances[cust,trip[i]] - data.distances[trip[i-1],trip[i]] + time_slack_to_add
    # time_slack_needed_current_trip = round(time_slack_needed_current_trip,5)

    # print("neeed time slack",time_slack_needed_current_trip)
    # print('travel dist', data.distances[trip[i-1],trip[i+1]])
    # time_slack_needed_next_trip = time_slack_needed_current_trip + parameters.DEPOT_LOADING_SCALAR*np.sum(data.demands[cust])
    # time_slack_needed_next_trip = time_slack_needed_current_trip + parameters.DEPOT_LOADING_SCALAR*data.service_time[cust]
    # print('Time adjustment',time_adjustment[k])
    # print('Time slack', time_slack[k][r][i])

    time_slack_avail = time_slack[k][r][i]
    # print("time_slack_avail_current_trip",time_slack_avail)
    if time_slack_needed_current_trip - time_slack_avail[1] > THRESHOLD:
        if log:
            logging.warning('Not enough time slack for current trip')
        return False
    
    
    return True


def route_check_single(route, data, initial_data, parameters,k,log=False):
    '''
    ### Input: Route from a single vehicle in [0,x,x,0,x,x,0] form (not split into trips yet)
'''
    route_validity = True
    

    # for k, route_veh in enumerate(routes):
    # route_veh = [cust_no_index(cust,data) for cust in route_veh]
    initial_route = [cust_no_index(cust,data) for cust in initial_data.initial_route[k]]
    # print(initial_route)
    # print('Route_veh:',route_veh)
    for z, initial_node in enumerate(initial_route): # check if route follows inital route
        if initial_node != route[z]:
            route_validity =False
            if log:
                logging.error("route does not follow inital route")
            return route_validity

    routes_trip = trip_split(route) #split into trips

    for r, route_trip in enumerate(routes_trip):
        if not (route_trip[-1] == 0): #route end in depot
            route_validity = False
            if log:
                logging.error("route does not end in depot")
            return route_validity



        ### CHECKING START AND END TIME WINDOWS
    time = initial_data.initial_time[k]
    load = np.array(initial_data.initial_loading[k])
    MAX_CAPACITY = tuple(parameters.MAX_CAPACITY)
    total_service_time = 0
    for r, route_trip in enumerate(routes_trip):
        # print ('trip:',r)
        prev_node = 0
        # if r != 0: #not the first trip, perfomr loading
        # print(MAX_CAPACITY)
        # total_to_load = np.array(MAX_CAPACITY)-np.array(load) #find amount of fuel to load
        loading_time = parameters.DEPOT_LOADING_SCALAR*np.sum(data.service_time[route_trip])
        total_service_time = 0
        time += loading_time #add loading time
        # print(time)
        load = np.array(MAX_CAPACITY) #set capacity back to maximum
            
        for a, node in enumerate(route_trip[1:]):
            if node == 0: # node is back at the depot
                travel_time = data.distances[prev_node,0] #travel back to depot
                time += travel_time
                # print (time)

                # ADDED THIS
                if time > data.end_time_windows[node]:
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

                if time > data.end_time_windows[node]:

                    route_validity =False
                    if log:
                        logging.error("route reach customer after end time window")
                    return route_validity
                # print(time)
                #calcualte service time and loading
                service_time = data.service_time[node]
                total_service_time += service_time
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


def route_check_trips(route, data, initial_data, parameters,k,log=False, THRESHOLD = 0.000001):
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
    total_service_time = 0
    for r, route_trip in enumerate(routes_trip):

        # print ('trip:',route_trip)
        prev_node = 0
        # if r != 0: #not the first trip, perfomr loading
        # print(MAX_CAPACITY)
        # total_to_load = np.array(MAX_CAPACITY)-np.array(load) #find amount of fuel to load
        loading_time = parameters.DEPOT_LOADING_SCALAR*np.sum(data.service_time[route_trip])
        total_service_time = 0
        time += loading_time #add loading time
        # print(time)
        load = np.array(MAX_CAPACITY) #set capacity back to maximum

        if len(route_trip) >2: #ignore empty trips
            for a, node in enumerate(route_trip[1:]):
                if node == 0: # node is back at the depot
                    travel_time = data.distances[prev_node,0] #travel back to depot
 
                    time += travel_time
                    #ADDED THIS
                    if time - data.end_time_windows[node]> THRESHOLD:
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

                    if time - data.end_time_windows[node] > THRESHOLD:
                        # print("Time",time)
                        # print("End time window",data.end_time_windows[node])
                        route_validity =False
                        if log:
                            logging.error("route reach customer after end time window")
                        return route_validity
                    #calcualte service time and loading

                    service_time = data.service_time[node]
                    time+=service_time
                    total_service_time += service_time
                    
                    
                    # print('time af', time)
                    
                    #calculate loading
                    service_load = np.array(data.demands[node])
                    # print("Cuirrent load", load)
                    # print("service load", service_load)
                    '''
                    if (service_load>MAX_CAPACITY).any():
                        route_validity =False
                        if log:
                            logging.error("Load at node exceeds vehicle capacity")
                        return route_validity'''
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
        total_service_time = 0
        for r, route_trip in enumerate(routes_trip):
            # print ('trip:',r)
            prev_node = 0
            # if r != 0: #not the first trip, perfomr loading
            # print(MAX_CAPACITY)
            # total_to_load = np.array(MAX_CAPACITY)-np.array(load) #find amount of fuel to load
            loading_time = parameters.DEPOT_LOADING_SCALAR*np.sum(data.service_time[route_trip])
            total_service_time = 0
            time += loading_time #add loading time
            # print(time)
            load = np.array(MAX_CAPACITY) #set capacity back to maximum
                
            for a, node in enumerate(route_trip[1:]):
                if node == 0: # node is back at the depot
                    travel_time = data.distances[prev_node,0] #travel back to depot
                    time += travel_time
                    # print (time)

                    #ADDED THIS
                    if time > data.end_time_windows[node]:
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
                    total_service_time += service_time
                    
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
    print('old', np.sum(data.demands[routes_flat])-parameters.ZETA*total_distance_travelled)
    print('new', data.total_revenue[routes_flat] -parameters.ZETA*total_distance_travelled)
    return data.total_revenue[routes_flat] -parameters.ZETA*total_distance_travelled


def objective_function(routes, data, initial_data, parameters):
    # total_demand_delivered = [Decimal(0)] * parameters.DEMAND_TYPES
    total_demand_delivered = [0] * parameters.DEMAND_TYPES
    total_cust_visited = 0
    # total_distance_travelled = Decimal(0)
    total_distance_travelled = 0
    for k, route_veh in enumerate(routes):
        # route_veh = [cust_no_index(cust,data) for cust in route_veh]
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
    # print ('demand:',sum(total_demand_delivered))
    # print ('cust:',total_cust_visited)
    # print ('dist:',total_distance_travelled)
    # print(parameters.ZETA)

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






'''
def objective_function_relaxed(routes, data, initial_data, parameters, ZETA = 0.1, PETA = 1):
    total_demand_delivered = np.zeros(parameters.DEMAND_TYPES)
    total_cust_visited = 0
    total_distance_travelled =0
    total_penalty = 0
    for k, route_veh in enumerate(routes):
        # route_veh = [cust_no_index(cust,data) for cust in route_veh]
        routes_trip = trip_split(route_veh) #split into trips
        time = initial_data.initial_time[k]
        load = np.array(initial_data.initial_loading[k])
        MAX_CAPACITY = tuple(parameters.MAX_CAPACITY)
        for r, route_trip in enumerate(routes_trip):
            prev_node = 0
            if r != 0: #not the first trip, perfomr loading
                # print(MAX_CAPACITY)
                total_to_load = np.array(MAX_CAPACITY)-np.array(load) #find amount of fuel to load
                loading_time = parameters.DEPOT_LOADING_SCALAR*sum(total_to_load)
                time += loading_time #add loading time
                # print(time)
                load = np.array(MAX_CAPACITY) #set capacity back to maximum

            for a, node in enumerate(route_trip[1:]):
                if node ==0:  # node is back at the depot
                    travel_time = data.distances[prev_node,0] #travel back to depot
                    time += travel_time
                    total_distance_travelled += data.distances[prev_node,0]
                else:
                    #travelling to next location
                    total_cust_visited+= 1
                    total_demand_delivered += data.demands[node]
                    total_distance_travelled += data.distances[prev_node,node]
                    travel_time = data.distances[prev_node,node]
                    time+= travel_time
                    #if reach before start time window, set time to start time window
                    if time < data.start_time_windows[node]:
                        time = data.start_time_windows[node]
                    # print(time)
                    #calcualte service time and loading
                    service_time = data.service_time[node]
                    time+=service_time
                    if time > data.end_time_windows[node]:
                        # print("Current Time:",time)
                        # print("End Time Window",data.end_time_windows[node])
                        total_penalty += time-data.end_time_windows[node]
                    #calculate loading
                    service_load = np.array(data.demands[node])
                    load -= service_load
                prev_node = node

    # print ('demand:',sum(total_demand_delivered))
    # print ('cust:',total_cust_visited)
    # print ('dist:',total_distance_travelled)

    return sum(total_demand_delivered)-ZETA*total_distance_travelled-PETA*total_penalty

def objective_function_relaxed_fast(routes, data, initial_data, parameters, ZETA = 0.1, PETA = 1,route_veh_dict = {}):
    # total_cust_visited = 0
    total_score = 0
    
    for k, route_veh in enumerate(routes):
        if tuple(route_veh) in route_veh_dict.keys():
            total_score += route_veh_dict[tuple(route_veh)]
            continue
        total_demand_delivered = np.zeros(parameters.DEMAND_TYPES)
        total_distance_travelled =0
        total_penalty = 0
        route_veh = [cust_no_index(cust,data) for cust in route_veh]
        routes_trip = trip_split(route_veh) #split into trips
        time = initial_data.initial_time[k]
        load = np.array(initial_data.initial_loading[k])
        MAX_CAPACITY = tuple(parameters.MAX_CAPACITY)
        for r, route_trip in enumerate(routes_trip):
            prev_node = 0
            if r != 0: #not the first trip, perfomr loading
                # print(MAX_CAPACITY)
                total_to_load = np.array(MAX_CAPACITY)-np.array(load) #find amount of fuel to load
                loading_time = parameters.DEPOT_LOADING_SCALAR*sum(total_to_load)
                time += loading_time #add loading time
                # print(time)
                load = np.array(MAX_CAPACITY) #set capacity back to maximum

            for a, node in enumerate(route_trip[1:]):
                if node ==0:  # node is back at the depot
                    travel_time = data.distances[prev_node,0] #travel back to depot
                    time += travel_time
                    total_distance_travelled += data.distances[prev_node,0]
                else:
                    #travelling to next location
                    # total_cust_visited+= 1
                    total_demand_delivered += data.demands[node]
                    total_distance_travelled += data.distances[prev_node,node]
                    travel_time = data.distances[prev_node,node]
                    time+= travel_time
                    #if reach before start time window, set time to start time window
                    if time < data.start_time_windows[node]:
                        time = data.start_time_windows[node]
                    # print(time)
                    #calcualte service time and loading
                    service_time = data.service_time[node]
                    time+=service_time
                    if time > data.end_time_windows[node]:
                        # print("Current Time:",time)
                        # print("End Time Window",data.end_time_windows[node])
                        total_penalty += time-data.end_time_windows[node]
                    #calculate loading
                    service_load = np.array(data.demands[node])
                    load -= service_load
                prev_node = node
        
        score = sum(total_demand_delivered)-ZETA*total_distance_travelled-PETA*total_penalty
        total_score += score
        route_veh_dict[tuple(route_veh)] = score
    # print ('demand:',sum(total_demand_delivered))
    # print ('cust:',total_cust_visited)
    # print ('dist:',total_distance_travelled)

    return total_score, route_veh_dict

from functools import cache

@cache
def route_score (routes_trip: tuple() ,time,load,distances,demands,service_time_data,start_time_windows,end_time_windows,\
                 DEMAND_TYPES,MAX_CAPACITY,DEPOT_LOADING_SCALAR):
    total_demand_delivered = np.zeros(DEMAND_TYPES)
    total_distance_travelled =0
    total_penalty = 0
    
    load = np.array(load)
    for r, route_trip in enumerate(routes_trip):
        prev_node = 0
        if r != 0: #not the first trip, perfomr loading
            # print(MAX_CAPACITY)
            total_to_load = np.array(MAX_CAPACITY)-np.array(load) #find amount of fuel to load
            loading_time = DEPOT_LOADING_SCALAR*sum(total_to_load)
            time += loading_time #add loading time
            # print(time)
            load = np.array(MAX_CAPACITY) #set capacity back to maximum

        for a, node in enumerate(route_trip[1:]):
            if node ==0:  # node is back at the depot
                travel_time = distances[prev_node][0] #travel back to depot
                time += travel_time
                total_distance_travelled += distances[prev_node][0]
            else:
                #travelling to next location
                # total_cust_visited+= 1
                total_demand_delivered += demands[node]
                total_distance_travelled += distances[prev_node][node]
                travel_time = distances[prev_node][node]
                time+= travel_time
                #if reach before start time window, set time to start time window
                if time < start_time_windows[node]:
                    time = start_time_windows[node]
                # print(time)
                #calcualte service time and loading
                service_time = service_time_data[node]
                time+=service_time
                if time > end_time_windows[node]:
                    # print("Current Time:",time)
                    # print("End Time Window",data.end_time_windows[node])
                    total_penalty += time-end_time_windows[node]
                #calculate loading
                service_load = np.array(demands[node])
                load -= service_load
            prev_node = node
    return sum(total_demand_delivered),total_distance_travelled,total_penalty


def objective_function_relaxed_fast_cache(routes, data, initial_data, parameters, ZETA = 0.1, PETA = 1,route_veh_dict = {}):
    # total_cust_visited = 0
    total_score = 0
    MAX_CAPACITY = tuple(parameters.MAX_CAPACITY)
    DEPOT_LOADING_SCALAR = parameters.DEPOT_LOADING_SCALAR
    DEMAND_TYPES = parameters.DEMAND_TYPES
    distances = tuple(tuple(i) for i in data.distances)
    demands = tuple(tuple(i) for i in data.demands)
    service_time_data = tuple(data.service_time)
    start_time_windows = tuple(data.start_time_windows)
    end_time_windows = tuple(data.end_time_windows)

    for k, route_veh in enumerate(routes):
        # route = test(route_veh,data,initial_data,parameters,k)
        time = initial_data.initial_time[k]
        load = tuple(initial_data.initial_loading[k])
        route_veh = [cust_no_index(cust,data) for cust in route_veh]
        routes_trip = trip_split(route_veh) #split into trips
        routes_trip = tuple(tuple(i) for i in routes_trip)
        total_demand_delivered,total_distance_travelled,total_penalty = route_score(tuple(routes_trip),time,load,distances,demands,service_time_data,start_time_windows,end_time_windows,\
                                                        DEMAND_TYPES,MAX_CAPACITY,DEPOT_LOADING_SCALAR)
        
        score = total_demand_delivered-ZETA*total_distance_travelled-PETA*total_penalty
        total_score += score
        # route_veh_dict[tuple(route_veh)] = score
    # print ('demand:',sum(total_demand_delivered))
    # print ('cust:',total_cust_visited)
    # print ('dist:',total_distance_travelled)

    return total_score, route_veh_dict
'''

if __name__ == '__main__':
    N = 100
    FILE = 'R201_ordered'
    N_VEHICLES = 3
    CAPACITY = 50
    DEMAND_TYPES = 5
    VEHICLE_COMPARMENTS = 5
    MAX_CAPACITY = np.array([CAPACITY]*DEMAND_TYPES)
    EMPTY_CAPACITY = np.array([0]*DEMAND_TYPES)
    data = Input.load_csv(FILE= FILE, N= N)
    parameters = InputParameters(N_VEHICLES=N_VEHICLES,DEMAND_TYPES=DEMAND_TYPES,VEHICLE_COMPARMENTS=VEHICLE_COMPARMENTS,
                                  DEPOT_LOADING_SCALAR=1,MAX_CAPACITY=MAX_CAPACITY,EMPTY_CAPACITY=EMPTY_CAPACITY)
    route1_cust_no = [1]
    route2_cust_no = [1]
    route3_cust_no = [1]

    time_start_1 = 0
    time_start_2 = 0
    time_start_3 = 0

    loading1 = MAX_CAPACITY
    loading2 = MAX_CAPACITY
    loading3 = MAX_CAPACITY

    initial_data = InitialData(
        initial_route=[route1_cust_no, route2_cust_no, route3_cust_no],
        initial_loading= [loading1,loading2,loading3],
        initial_time= [time_start_1,time_start_2,time_start_3]
    )
    route1 = [1,36,2,1,90,61,18,94,101,59,1]
    route2 = [1,75,1,78,81,1]
    route3 = [1,14,71,69,1]

    routes = [route1,route2,route3]
    # routes = [[1, 34, 40, 16, 15, 99, 3, 22, 24, 1], [1, 73, 29, 28, 70, 53, 32, 12, 83, 1], [1, 43, 93, 60, 96, 6, 46, 37, 48, 63, 65, 1]]
    routes = [[1,60,93,96,3,29,28,16,19,58,1,97,38,57,56,55,5,75,101,92,18,61,90,1],
              [1,73,43,15,46,37,48,24,89,8,1,11,51,21,33,14,1,59,94,1],
              [1,34,66,64,83,84,32,63,65,20,1,51,4,35,27,1,2,36,25,26,81,78,71,1]]
    print (route_check(routes,data,initial_data,parameters))
    # print (objective_function(routes,data,initial_data,parameters))

    
    def flatten_comprehension(matrix):
        return [item for row in matrix for item in row]
    cust_visted = flatten_comprehension(routes)
    cust_visted = [i for i in cust_visted if i!=1]
    print("Customers Visited:", len(cust_visted))
