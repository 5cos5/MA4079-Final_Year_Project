from read_data import Input, InitialData, InputParameters
from model_check import cust_no_index, trip_split, route_check,objective_function
from trip_generator import create_trips, remove_trips, remove_depots
import numpy as np
import matplotlib.pyplot as plt
import logging

def plot_route(routes: list[list],data,initial_data,parameters):
    '''
    Given the input of routes taken, plots the route taken into a matplotlib plot
    '''
    coordinates = data.coordinates
    cust_no = data.cust_no
    N_VEHICLES = len(routes)
    # Choosing colors
    colors = ["red","blue","green",'orange','purple','yellow','black','black','black','black']

    fig, ax = plt.subplots()
    # ax.set_aspect('equal')

    # Now the figure
    for i, coord in enumerate(coordinates):
        x = coord[0]
        y= coord[1]
        ax.scatter(x,y, color = 'grey')
        ax.annotate(cust_no[i], (x,y))

    #print ('Number of nodes:',N)
    for k, route_veh in enumerate(routes):
        c = colors[k]
        # route_veh = [cust_no_index(cust,data) for cust in route_veh]
        prev_node = route_veh[0]
        for a, node in enumerate(route_veh[1:]):
            coord1 = coordinates[prev_node]
            coord2 = coordinates[node]
            x1 = coord1[0]
            y1 = coord1[1]
            x2 = coord2[0]
            y2 = coord2[1]
            '''
            service_load = np.array(data.demands[node])
            f= np.nonzero(service_load)
            d = service_load[f].astype(int)[0]
            f = f[0][0]
            '''
            ax.scatter(x1,y1, color = c)
            
            # ax.annotate((i,d), (x1,y1))
            ax.plot((x1,x2), (y1,y2), color=c, label= k+1)
            prev_node = node
    # plt.show()
#ax.legend()
'''
for k in range(N_VEHICLES):
    print("\nVehicle no:", k+1)
    node = 0
    #print (node, end ='')
    for r in model.R:
        for i in range(len(demands)):
            for j in range(len(demands)):
                    if node == j:
                        continue
                    elif np.isclose(model.X[node, j, k,r].value, 1, atol=1e-1):
                        print (cust_no[node], " --", cust_no[j], end =' ')
                        print ("R:",r)
                        node = j
                        break
            if node == 0:
                break
    #print ("\n")
'''
def routes_time(routes, data, initial_data, parameters):
    total_demand_delivered = 0
    total_cust_visited = 0
    total_distance_travelled =0
    time_array = []
    
    for k, route_veh in enumerate(routes):
        time_list = []
        # route_veh = [cust_no_index(cust,data) for cust in route_veh]
        # print(route_veh)
        routes_trip = trip_split(route_veh) #split into trips
        
        time = initial_data.initial_time[k]
        load = np.array(initial_data.initial_loading[k])
        MAX_CAPACITY = tuple(parameters.MAX_CAPACITY)
        for r, route_trip in enumerate(routes_trip):
            
            prev_node = 0
            if r == 0:
                time_list.append((0,time,time))
            if r != 0: #not the first trip, perfomr loading
                total_to_load = MAX_CAPACITY-load #find amount of fuel to load
                loading_time = parameters.DEPOT_LOADING_SCALAR*sum(total_to_load)
                time += loading_time #add loading time
                load = MAX_CAPACITY #set capacity back to maximum
                time_list[-1][-1] = time
            for a, node in enumerate(route_trip[1:]):
                if node ==0:  # node is back at the depot
                    travel_time = data.distances[prev_node,0] #travel back to depot
                    time += travel_time
                    time_list.append([node,time,time])
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
                    time_list.append((node,time))
                    #calcualte service time and loading
                    
                    if time > data.end_time_windows[node]:
                        route_validity =False
                        logging.error("In final route reach customer after end time window")
                    service_time = data.service_time[node]
                    time+=service_time
                    #calculate loading
                    service_load = np.array(data.demands[node])
                    load -= service_load
                prev_node = node
        time_array.append(time_list)

    # print ('demand:',sum(total_demand_delivered))
    # print ('cust:',total_cust_visited)
    # print ('dist:',total_distance_travelled)
    print (time_array)
    return time_array

def plot_time(routes: list[list],data,initial_data,parameters):
    """
    Given the route, plot the schedule of each barge
    """
    N_VEHICLES = parameters.N_VEHICLES
    START_TIME = min(data.start_time_windows)
    END_TIME = max(data.end_time_windows)
    # coordinates = data.coordinates
    cust_no = data.cust_no

    fig, ax = plt.subplots(figsize = (5,20))
    ax.set_xlim(0, N_VEHICLES + 2)
    ax.set_ylim(START_TIME, END_TIME)
    ax.set_xticks(np.arange(1, N_VEHICLES + 1))
    time_array = routes_time(routes,data,initial_data,parameters)
    for k, route_veh in enumerate(routes):
        ax.plot([k+1, k+1], [START_TIME,END_TIME], linewidth=2, color='b')  #vessel docking period
        # route_veh = [cust_no_index(cust,data) for cust in route_veh]
        # print(route_veh)
        # prev_node = 0
        time = 0
        for a, node in enumerate(route_veh):
            # print('k',k)
            # print('a',a)
            time_start_at_node = time_array[k][a][1]
            # print(time_start_at_node,node)
            time = time_start_at_node

            if node !=0: #servicing customers
                service_time_at_node = data.service_time[node]
                ax.add_patch(plt.Rectangle((k+1-0.25, time), 0.5, service_time_at_node, edgecolor='black',\
                                linewidth=2,facecolor='none'))
                ax.text(k+1-0.125, time+(service_time_at_node/2), cust_no[node], color = 'black', fontsize = 8)
            else: #starting at terminal
                if a != 0: #not the start of the trip
                    service_time_at_node = time_array[k][a][2] - time_array[k][a][1]
                else: #start of the trip, assume no servicing at depot
                    service_time_at_node = 0 
                ax.add_patch(plt.Rectangle((k+1-0.25, time), 0.5, service_time_at_node, edgecolor='red',\
                                linewidth=2,facecolor='none'))
            # print("service",service_time_at_node)
            time+= service_time_at_node

            try:
                next_node = route_veh[a+1]
            except:
                next_node = 0
            
            travelling_time = data.distances[node,next_node]
            # print('travel',travelling_time)
            if travelling_time != 0:
                ax.add_patch(plt.Rectangle((k+1-0.25, time), 0.5, travelling_time, edgecolor='orange',\
                            linewidth=2,facecolor='none'))
            
            # time+=travelling_time

            # prev_node = node
    # plt.show()
if __name__ == '__main__':
    N = 100
    N_VEHICLES = 3
    # FILE = 'R201_ordered'
    FILE = 'shuffled_ordered_data/R211_shuffled_ordered'
    CAPACITY = 50
    DEMAND_TYPES = 5
    VEHICLE_COMPARMENTS = 5
    MAX_CAPACITY = np.array([CAPACITY]*DEMAND_TYPES)
    EMPTY_CAPACITY = np.array([0]*DEMAND_TYPES)
    ZETA = 0.1
    parameters = InputParameters(N_VEHICLES=N_VEHICLES,DEMAND_TYPES=DEMAND_TYPES,VEHICLE_COMPARMENTS=VEHICLE_COMPARMENTS,
                                DEPOT_LOADING_SCALAR=1,MAX_CAPACITY=MAX_CAPACITY,EMPTY_CAPACITY=EMPTY_CAPACITY, ZETA=ZETA)
    data = Input.load_csv(FILE= FILE, N= N, parameters= parameters)
    
    route1_cust_no = []
    route2_cust_no = []
    route3_cust_no = []

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
    route1 = [1, 40, 96, 93, 64, 48, 1, 60, 28, 29, 84, 6, 43, 3, 1, 32, 1]
    route2 = [1, 1]
    route3 = [1, 93, 64, 48, 1, 60, 28, 29, 84, 6, 43, 3, 1, 32, 73, 1]

    # routes =[[1, 30, 16, 6, 1], [1, 17, 24, 19, 23, 1], [1, 14, 1]]
    # routes =[[1, 30, 16, 6, 1, 20, 15, 10, 21, 26, 1], [1, 17, 24, 19, 23, 1, 11, 28, 29, 25, 4, 9, 1], [1, 31, 12, 13, 2, 8, 3, 18, 1, 14, 1]]
    # routes = [[1, 34, 13, 43, 45, 3, 1, 88, 62, 84, 93, 19, 1, 96, 89, 64, 1], [1, 28, 53, 24, 68, 1, 46, 16, 39, 17, 74, 76, 1, 30, 12], [1, 99, 32, 66, 72, 1, 77, 29, 22, 41, 31, 63, 8, 83, 20, 1]]
    # routes = ((1, 40, 3, 96, 28, 32, 63, 12, 65, 1), (1, 66, 34, 29, 73, 15, 16, 24, 1), (1, 93, 60, 6, 46, 37, 48, 83, 99, 53, 70, 1))
    routes = [[1, 89, 63, 65, 66, 80, 77, 96, 93, 45, 39, 43, 88, 3, 41, 1], [1, 22, 74, 73, 76, 24, 16, 17, 99, 100, 84, 46, 37, 48, 20, 64, 52, 1], [1, 29, 13, 34, 31, 70, 28, 1, 82, 32, 53, 8, 83, 19, 6, 62, 87, 86, 1]]
    # print (tuple(routes))
    print(routes_time(routes,data,initial_data,parameters))
    print(route_check(routes,data,initial_data,parameters,log =True))
    
    plot_time(routes,data,initial_data,parameters)
    plot_route(routes,data,initial_data,parameters)
    
    plt.show()
    