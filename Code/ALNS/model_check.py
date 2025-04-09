from read_data import Input, InitialData, InputParameters
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



def find_time_slack_newest(routes_trips,data,initial_data,parameters,log =False):
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
        # flat_time_slacks_list = [item for row in time_slacks_list[k] for item in row]
        # new_flat_time_adustment_list = [item for row in time_adjustment_list[k][:-1] for item in row]
        # new_flat_time_slacks_list = [item for row in time_slacks_list[k][:-1] for item in row]

        newer_flat_time_adustment_list = [item for row in time_adjustment_list[k] for item in row]
        newer_flat_time_slacks_list = [item for row in time_slacks_list[k] for item in row]

        # print('flat_time_adustment_list',flat_time_adustment_list)
        time_adj = time_adjustment_list[k]
        # temp_all_prev_trips_list =np.array([x+sum(flat_time_adustment_list[:y+1]) for y, x in enumerate(flat_time_slacks_list[:-2])]) #ignore the last 2 locations
        # temp_all_prev_trips_list_new = np.array([x+sum(new_flat_time_adustment_list[:y+1]) for y, x in enumerate(new_flat_time_slacks_list)])
        
        temp_all_prev_trips_list_newer = np.array([x+sum(newer_flat_time_adustment_list[:y+1]) for y, x in enumerate(newer_flat_time_slacks_list)])
        # assert temp_all_prev_trips_list.any() == temp_all_prev_trips_list_new.any()
        count_flat = len(time_slacks_veh[0])
        # print('time', time_list[k])
        # print('time_slacks_veh',time_slacks_veh)
        # print(temp_all_prev_trips_list)

        for r, time_slacks_trip in enumerate(time_slacks_veh):
            trip_length = len(time_slacks_trip)
            # time_slacks_trip_fixed = [time_slacks_trip[0]]
            time_slacks_trip_fixed = []
           

            if r < len(time_slacks_veh) -2: #ignore the last trip and dummy last
                # flat_next_time_slacks_veh = [item for row in time_slacks_veh[r+1:-1] for item in row] #ignore dummy last
                # flat_next_time_adjustment_veh = [item for row in time_adjustment_list[k][r+1:-1] for item in row] #ignore dummy last
                # print(time_adjustment_list[k])
                # print(flat_next_time_adjustment_veh)
                # temp_only_next_trips = [x+sum(flat_next_time_adjustment_veh[:y+1]) for y, x in enumerate(flat_next_time_slacks_veh)]

                # temp6 = round(min(temp_only_next_trips),5)
                temp_all_prev_trips_list_newer_temp6 = temp_all_prev_trips_list_newer[trip_length:-2] - sum(time_adjustment_list[k][r])
                # temp6 = round(min(temp_all_prev_trips_list_newer_temp6),5)
                temp6 = min(temp_all_prev_trips_list_newer_temp6)
                # assert temp6 == temp6_new
            else: # this is not important as it wont be checked
                # temp6 = round(time_slacks_trip[-1],5)
                temp6 = -1
            
            # temp_only_current_trip_list = np.array([x+sum(time_adjustment_list[k][r][:y+1]) for y, x in enumerate(time_slacks_veh[r])])
            for i, time_slack in enumerate(time_slacks_trip):
                # if i != 0:
                    time_slacks_node_fixed = []
                    # print(i)
                    # print('Time adjustment',time_adjustment_list[k][r])
                    # print('Time slacks',time_slacks_veh[r])
                    # print('Time', time_list[k][r])
                    # temp_only_current_trip = [x+sum(time_adjustment_list[k][r][i:i+y+1]) for y, x in enumerate(time_slacks_veh[r][i:])]
                    
                    # print('current_trip',temp_only_current_trip )
                    # print('list', temp_only_current_trip_list[i:])
                    
                    # temp1 = round(min(temp_all_prev_trips),5)
                    # temp2 = round(min(temp_only_current_trip),5)

                    # temp3 = round(min(temp_only_current_trip_list[i:]),5)
                    # temp3 = round(min(temp_all_prev_trips_list_newer[i:trip_length]),5)
                    temp3 = min(temp_all_prev_trips_list_newer[i:trip_length])

                    # print('temp_only_current_trip_list',temp_only_current_trip_list[i:])
                    
                    # print('temp_all_prev_trips_list', temp_all_prev_trips_list[count_flat:])
       
                    # print('count_flat', count_flat)
                    if r < len(time_slacks_veh)-2: #ignore the last trip and dummy last
                        # temp5 = round(min(temp_all_prev_trips_list_new[count_flat:]),5)
                        # temp5 = round(min(temp_all_prev_trips_list_newer[trip_length:-2]),5)
                        temp5 = min(temp_all_prev_trips_list_newer[trip_length:-2])
                        # assert temp5 == temp5_new
                    else: # this is not impt as it wont be checked
                        # temp5 = round(temp_all_prev_trips_list_new[-1],5)
                        # temp5_new = round(temp_all_prev_trips_list_newer[-1],5)
                        temp5 = -1
                        # print('fail')
                        # print('temp5',temp_all_prev_trips_list_new)
                        # print('temp5_new',temp_all_prev_trips_list_newer)
                        # assert temp5 == temp5_new

                    # time_slacks_node_fixed.append((temp3,temp5,temp6))
                    temp_all_prev_trips_list_newer -= time_adjustment_list[k][r][i]
                    # temp_all_prev_trips_list_new -= time_adjustment_list[k][r][i]
                    # temp_only_current_trip_list -= time_adjustment_list[k][r][i]
                    
                    time_slacks_trip_fixed.append((temp3,temp5,temp6))
            # count_flat += len(time_slacks_trip)
            # pointer += trip_length
            temp_all_prev_trips_list_newer = temp_all_prev_trips_list_newer[trip_length:]

            if r < len(time_slacks_veh)-2:
                count_flat += len(time_slacks_veh[r+1])

            time_slacks_veh_fixed.append(time_slacks_trip_fixed)
        time_slacks_fixed.append(time_slacks_veh_fixed)
    # print('Fixed Time Slack', time_slacks_fixed)
    # raise Exception

    return time_list, time_slacks_fixed,time_adjustment_list,loading_slack_list

def check_insertion_linear_newest(time,time_slack,time_adjustment,demand_slack,route,trip,k,r,i,cust,data,initial_data,parameters,log = False, THRESHOLD = 0.000001):
    # print('Demand slack:',type(demand_slack[k][r]))
    # print('Demand',data.demands[cust])
    # print('Difference',demand_slack[k][r]-data.demands[cust])
    # if np.any(demand_slack[k][r]-data.demands[cust]<parameters.EMPTY_CAPACITY):
    if (demand_slack[k][r]-data.demands[cust]<parameters.EMPTY_CAPACITY).any():
        # print('demand slack', demand_slack[k][r])
        # if log:    
            # logging.warning("Over Demand")
        return False

    # print('rouite', route)
    # print('trip', trip)
    # print('time',time[k][r])
    # print('position kri',(k,r,i))
    # print('inserted_cust', cust)
    time_visit = time[k][r][i-1]
    # print('B4 time',time_visit)
    time_visit += data.service_time[trip[i-1]]
    time_visit += data.distances[trip[i-1],cust]
    # print('Af time', time_visit)
    time_slack_to_add = 0
    # print(time_visit)
    if time_visit < data.start_time_windows[cust]:
        # print("START_TIME_WINDOWS", data.start_time_windows[cust])
        time_slack_to_add = data.start_time_windows[cust] - time_visit
        # print('TIME SLACK ADDED',time_slack_to_add )
    if time_visit - data.end_time_windows[cust] > THRESHOLD:
        # print(time)
        # print(trip)
        # print("i",i)
        # print("Visit time",time_visit)
        # print("End time window",data.end_time_windows[cust])
        # if log:
            # logging.warning("Newest Time past time windows")
        return False
    time_slack_needed_current_trip = data.service_time[cust] + data.distances[trip[i-1],cust] + data.distances[cust,trip[i]] - data.distances[trip[i-1],trip[i]] + time_slack_to_add
    # time_slack_needed_current_trip = round(time_slack_needed_current_trip,5)

    # print("neeed time slack",time_slack_needed_current_trip)
    # print('travel dist', data.distances[trip[i-1],trip[i+1]])
    # time_slack_needed_next_trip = time_slack_needed_current_trip + parameters.DEPOT_LOADING_SCALAR*np.sum(data.demands[cust])
    # time_slack_needed_next_trip = round(time_slack_needed_next_trip,5)
    # print('Time adjustment',time_adjustment[k])
    # print('Time slack', time_slack[k][r])
    time_slack_avail_current_trip = time_slack[k][r][i]
    # print("time_slack_avail_current_trip",time_slack_avail_current_trip)
    # print("time_slack_needed_next_trip", time_slack_needed_next_trip)
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
        for r, time_slacks_trip in enumerate(time_slacks_veh):
            time_slacks_trip_fixed = [time_slacks_trip[0]]
            for i, time_slack in enumerate(time_slacks_trip):
                if i != 0:
                    time_slacks_node_fixed = []
                    # print(i)
                    # print('Time adjustment',time_adjustment_list[k][r])
                    # print('Time slacks',time_slacks_veh[r])
                    # print('Time', time_list[k][r])
                    temp_only_current_trip = [x+sum(time_adjustment_list[k][r][i:i+y+1]) for y, x in enumerate(time_slacks_veh[r][i:])]
                    # print('current_trip',temp_only_current_trip )
                    
                    # temp1 = round(min(temp_all_prev_trips),5)
                    temp2 = round(min(temp_only_current_trip),5)
                    time_slacks_node_fixed.append((temp2))
                    
                    for idx, _ in enumerate(time_slacks_veh[:-1]): #ignore the last empty trip
                        if idx <= r: #skip trips b4 the one we are analysing
                            continue
                        else:
                            # print('Time adjustment',time_adjustment_list[k][idx])
                            # print('flat_time_adustment_list',flat_time_adustment_list[(r+1)*(i):])
                            # print('Time slacks',time_slacks_veh[idx])
                            # print('Time', time_list[k][idx])
                            time_adj = time_adjustment_list[k]
                            # print(time_adj)
                            # print([adj for a in time_adj[r:idx+1] for adj in a])
                            # temp_all_prev_trips = [x+sum([adj for a in time_adj[r:idx+1] for adj in a][i:y+1]) for y,x in enumerate(time_slacks_veh[idx])]
                            temp_all_prev_trips = [x+sum([adj for a in time_adj[r:idx+1] for adj in a][i:][::-1][y:]) for y,x in enumerate(time_slacks_veh[idx][::-1])]
                            # temp_all_prev_trips = [x+sum(flat_time_adustment_list[(r+1)*(i):(idx+1)*(y+1)]) for y, x in enumerate(time_slacks_veh[idx][:])]
                            # print('All prev trips',temp_all_prev_trips)
                            temp_only_next_trips = [x+sum([adj for a in time_adj[r+1:idx+1] for adj in a][::-1][y:]) for y,x in enumerate(time_slacks_veh[idx][::-1])]
                            # temp_only_next_trips = [x+sum(time_adjustment_list[k][idx][:y+1]) for y, x in enumerate(time_slacks_veh[idx][:])]
                            # print('next_trips',temp_only_next_trips )
                            
                            temp1 = round(min(temp_all_prev_trips),5)
                            temp2 = round(min(temp_only_next_trips),5)
                            time_slacks_node_fixed.append((temp1,temp2))
                
                    time_slacks_trip_fixed.append(time_slacks_node_fixed)
                
            time_slacks_veh_fixed.append(time_slacks_trip_fixed)
        time_slacks_fixed.append(time_slacks_veh_fixed)
    # print('Fixed Time Slack', time_slacks_fixed)
    # raise Exception

    return time_list, time_slacks_fixed,time_adjustment_list,loading_slack_list
                
def check_insertion_linear_new(time,time_slack,time_adjustment,demand_slack,route,trip,k,r,i,cust,data,initial_data,parameters,log = False):
    # print(demand_slack[k][r])
    if (demand_slack[k][r]-data.demands[cust]<parameters.EMPTY_CAPACITY).any():
        # print('demand slack', demand_slack[k][r])
        if log:    
            logging.warning("Over Demand")
        return False

    # print('rouite', route)
    # print('trip', trip)
    # print('time',time)
    # print('inserted_cust', cust)
    time_visit = time[k][r][i-1]
    # print('B4 time',time_visit)
    time_visit += data.service_time[trip[i-1]]
    time_visit += data.distances[trip[i-1],cust]
    # print('Af time', time_visit)
    time_slack_to_add = 0
    # print(time_visit)
    if time_visit < data.start_time_windows[cust]:
        # print("START_TIME_WINDOWS", data.start_time_windows[cust])
        time_slack_to_add = data.start_time_windows[cust] - time_visit
        # print('TIME SLACK ADDED',time_slack_to_add )
    if time_visit > data.end_time_windows[cust]:
        # print(time)
        # print(trip)
        # print("i",i)
        # print("Visit time",time_visit)
        # print("End time window",data.end_time_windows[cust])
        if log:
            logging.warning("Time past time windows")
        return False
    time_slack_needed_current_trip = data.service_time[cust] + data.distances[trip[i-1],cust] + data.distances[cust,trip[i+1]] - data.distances[trip[i-1],trip[i+1]] + time_slack_to_add
    time_slack_needed_current_trip = round(time_slack_needed_current_trip,5)

    # print("neeed time slack",time_slack_needed_current_trip)
    # print('travel dist', data.distances[trip[i-1],trip[i+1]])
    time_slack_needed_next_trip = time_slack_needed_current_trip + parameters.DEPOT_LOADING_SCALAR*np.sum(data.demands[cust])
    # print('Time adjustment',time_adjustment[k])
    # print('Time slack', time_slack[k][r])
    time_slack_avail_current_trip = time_slack[k][r][i][0]
    # print("time_slack_avail_current_trip",time_slack_avail_current_trip)
    if time_slack_needed_current_trip > time_slack_avail_current_trip:
        if log:
            logging.warning('Not enough time slack for current trip')
        return False
    
    if r < len(route)-2: # if the trip we are analysing is not the last trip b4 dummy
        # print('NExt trip')
        for idx, time_slack_avail_next_trip in enumerate(time_slack[k][r][i][1:]):
            # print(time_adjustment)
            # print ("Time slacks for next trip", time_slack[k][r][i][1:])
            # print("time_slack_avail_next_trip",time_slack_avail_next_trip)
            # print("neeed time slack",time_slack_needed_next_trip)
            if time_slack_needed_next_trip > time_slack_avail_next_trip[0]:
                if log:
                    logging.warning('Not enough time slack for next trip')
                return False
            
            if parameters.DEPOT_LOADING_SCALAR*np.sum(data.demands[cust]) > time_slack_avail_next_trip[1]:
                if log:
                    logging.warning('Not enough time slack for next trip due to loading')
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
    # print ('demand:',(total_demand_delivered*parameters.PRICE))
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

### FUNCTIONS BELOW THIS IS NOT USED
def check_insertion_linear(time,time_slack,time_adjustment,same_time_slack,demand_slack,route,trip,k,r,i,cust,data,initial_data,parameters,log = False):
    if (demand_slack[k][r]-data.demands[cust]<parameters.EMPTY_CAPACITY).any():
        if log:    
            logging.warning("Over Demand")
        return False
    
    print('rouite', route)
    print('trip', trip)
    print('time',time)
    print('inserted_cust', cust)
    time_visit = time[k][r][i-1]
    # print('B4 time',time_visit)
    time_visit += data.service_time[trip[i-1]]
    time_visit += data.distances[trip[i-1],cust]
    # print('Af time', time_visit)
    time_slack_to_add = 0
    # print(time_visit)
    if time_visit < data.start_time_windows[cust]:
        # print("START_TIME_WINDOWS", data.start_time_windows[cust])
        time_slack_to_add = data.start_time_windows[cust] - time_visit
        # print('TIME SLACK ADDED',time_slack_to_add )
    if time_visit > data.end_time_windows[cust]:
        # print(time)
        # print(trip)
        # print("i",i)
        # print("Visit time",time_visit)
        # print("End time window",data.end_time_windows[cust])
        if log:
            logging.warning("Time past time windows")
        return False
    
    time_slack_needed_current_trip = data.service_time[cust] + np.sum(data.distances[[trip[i-1],cust],[cust,trip[i+1]]]) - data.distances[trip[i-1],trip[i+1]] + time_slack_to_add
    # print("neeed time slack",time_slack_needed_current_trip)
    # print('travel dist', data.distances[trip[i-1],trip[i+1]])
    time_slack_needed_next_trip = time_slack_needed_current_trip + parameters.DEPOT_LOADING_SCALAR*np.sum(data.demands[cust])
     
    
    min_time_slack = min(time_slack[k][r][i:])
    '''
    print('Time slack avail trip:', min_time_slack)
    print('Time slack',time_slack)
    print("time_adjustment", time_adjustment)
   
    if time_slack_needed_current_trip > min_time_slack:
        min_time_slack = round(min_time_slack,5)
        if time_slack_needed_current_trip > min_time_slack:
            return False'''

    if time_slack_needed_current_trip > min_time_slack:
        min_time_slack_temp = round(min_time_slack,5) # due to precision error, do some rounding

        if time_slack_needed_current_trip > min_time_slack_temp:
            ### Add any waiting time between added cust and cust with least time slack
            # min_time_slack = time_slack[k][r][i]
            res = len(time_slack[k][r]) - 1 - time_slack[k][r][::-1].index(min_time_slack)
            to_also_check = same_time_slack[k][r][i:res]
            res_list= [j for j in range(i,res+1)]
            # res_list.extend([j+i for j, x in enumerate(to_also_check) if x])
            
            for res in res_list:
                time_adjustment_trip = sum(time_adjustment[k][r][i:res+1])
                time_slack_needed_current_trip_temp = time_slack_needed_current_trip
                time_slack_needed_current_trip_temp -= time_adjustment_trip
                print("neeed time slack",time_slack_needed_current_trip_temp)
                print(res)
                print("time_adjustment", time_adjustment)
                print('Time slack',time_slack)
                print('Time slack avail trip:', min_time_slack)
                print('Critical customer', trip[res+1])
                if time_slack_needed_current_trip_temp > min_time_slack:
                    if log:
                        logging.warning('Not enough time slack')
                    return False
    
    for idx, next_time_slack in enumerate(time_slack[k][r+1:-1]): #ignore the additional trip if it is not used
        
        if time_slack_needed_next_trip > next_time_slack[0]:
            min_time_slack_next_trip_round = round(next_time_slack[0],5) # due to precision error, do some rounding
            if time_slack_needed_next_trip > min_time_slack_next_trip_round:
                # print('Preliminary next trip failed')
                ### Add any waiting time till cust with least time slack
                min_time_slack_next_trip = next_time_slack[0]
                res = len(next_time_slack) - 1 - next_time_slack[::-1].index(min_time_slack_next_trip)
                time_adjustment_trip = sum(time_adjustment[k][r][i:])
                time_slack_needed_next_trip -= time_adjustment_trip

                to_also_check = same_time_slack[k][r+1+idx][i:res]
                res_list= [res]
                res_list.extend([j+i for j, x in enumerate(to_also_check) if x])
                if len(res_list) >1:
                    print(res_list)
                    '''
                    ### THis is not verified to be working
                    '''
                    raise Exception('Multiple places to check')
                for res in res_list:

                    time_adjustment_next_trip = sum(time_adjustment[k][r+1+idx][:res+1])
                    time_slack_needed_next_trip_temp = time_slack_needed_next_trip
                    time_slack_needed_next_trip_temp -= time_adjustment_next_trip
                    

                    # print("Next trip neeed time slack",time_slack_needed_next_trip)
                    # print('Time slack avail in next trip',min_time_slack_next_trip_round)
                    if time_slack_needed_next_trip_temp > min_time_slack_next_trip_round:
                        if log:
                            logging.warning('Not enough time slack in next trip')
                        return False
                    ## Check the case where additional time taken to top up results in failure of next trip
                    if parameters.DEPOT_LOADING_SCALAR*np.sum(data.demands[cust]) > min_time_slack_next_trip - time_adjustment_next_trip:
                        if log:
                            logging.warning('Not enough time slack in next trip due to top up')
                        return False
    '''
    try:
        # print("Next trip neeed time slack",time_slack_needed_next_trip)
        # print('Time slack avail in next trip',time_slack[k][r+1][0])
        #check general case where next trip fails
        if time_slack_needed_next_trip > time_slack[k][r+1][0]:
            # print('Preliminary next trip failed')
            ### Add any waiting time till cust with least time slack
            min_time_slack_next_trip = time_slack[k][r+1][0]
            res = len(time_slack[k][r+1]) - 1 - time_slack[k][r+1][::-1].index(min_time_slack_next_trip)
            time_adjustment_trip = sum(time_adjustment[k][r][i:])
            time_adjustment_next_trip = sum(time_adjustment[k][r+1][:res])
            time_slack_needed_next_trip -= time_adjustment_next_trip
            time_slack_needed_next_trip -= time_adjustment_trip

            if time_slack_needed_next_trip > min_time_slack_next_trip:
                if log:
                    logging.warning('Not enough time slack in next trip')
                return False
            ## Check the case where additional time taken to top up results in failure of next trip
            if parameters.DEPOT_LOADING_SCALAR*np.sum(data.demands[cust]) > min_time_slack_next_trip - time_adjustment_next_trip:
                if log:
                    logging.warning('Not enough time slack in next trip')
                return False
    except: # no more next trip
        pass
    '''

    return True



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
