
from read_data_expt2 import Input, InitialData, InputParameters
from model_check import route_check, objective_function,cust_no_index,objective_function_fast,trip_split,find_time_slack_new,route_check_single,route_check_trips
# from greedy_insertion import initial_greedy_insertion_distance
from initial import initial_greedy_insertion_distance
from destroy import random_remove_point,greedy_remove_distance,shaw_removal
from repair import random_cust_best_insertion_fastest,best_regret2_fastest,best_insertion_fastest
from trip_generator import create_trips, remove_trips, remove_depots,remove_repeated_depot
from plot_data import plot_route, plot_time
# from route import Routes
from plot_data import plot_route, plot_time
import numpy as np
import timeit
from tqdm import tqdm
from math import ceil
import matplotlib.pyplot as plt
import copy
N = 100
# FILE = 'R201_ordered'
# FILE = 'shuffled_ordered_data/R211_shuffled_ordered'
FILE = 'experiment2_data/RC201_test'
# FILE = 'experiment2_data/C201_test'
N_VEHICLES = 8
CAPACITY = 100
DEMAND_TYPES = 5
VEHICLE_COMPARMENTS = 5
MAX_CAPACITY = np.array([CAPACITY]*DEMAND_TYPES)
EMPTY_CAPACITY = np.array([0]*DEMAND_TYPES)
ZETA = 0.01
parameters = InputParameters(N_VEHICLES=N_VEHICLES,DEMAND_TYPES=DEMAND_TYPES,VEHICLE_COMPARMENTS=VEHICLE_COMPARMENTS,
                                DEPOT_LOADING_SCALAR=0.2,MAX_CAPACITY=MAX_CAPACITY,EMPTY_CAPACITY=EMPTY_CAPACITY, ZETA=ZETA)
data = Input.load_csv(FILE= FILE, N= N, parameters= parameters)
route1_cust_no = [1]
route2_cust_no = [1]
route3_cust_no = [1]

time_start_1 = 0
time_start_2 = 0
time_start_3 = 0

loading1 = [CAPACITY,CAPACITY,CAPACITY,CAPACITY,CAPACITY]
loading2 = [CAPACITY,CAPACITY,CAPACITY,CAPACITY,CAPACITY]
loading3 = [CAPACITY,CAPACITY,CAPACITY,CAPACITY,CAPACITY]

initial_data = InitialData(
    initial_route=[route1_cust_no, route2_cust_no],# route3_cust_no],
    initial_loading= [loading1,loading2,loading3],
    initial_time= [time_start_1,time_start_2,time_start_3]
)

initial_data = InitialData(
    initial_route=[route1_cust_no] *N_VEHICLES,
    initial_loading= [loading1] *N_VEHICLES,
    initial_time= [time_start_1]*N_VEHICLES
)


routes_cust =  [[1, 21, 23, 25, 7, 24, 1, 19, 20, 17, 15, 13, 1, 26, 12, 11, 9, 22, 1], [1, 6, 3, 2, 8, 4, 5, 1, 18, 16, 14, 10, 1]]
routes = []
for route in routes_cust:
    routes.append([cust_no_index(i,data) for i in route])
# print(routes_cust)
# routes =  [[0, 24, 6, 29, 4, 46, 11, 14, 0, 37, 34, 47, 21, 49, 0], [0, 1, 23, 15, 25, 43, 31, 30, 22, 9, 26, 35, 19, 40, 16, 0], [0, 32, 8, 17, 0, 36, 44, 39, 10, 27, 5, 13, 50, 33, 41, 48, 0]]
# routes = [[0, 24, 1, 23, 5, 13, 0, 46, 32, 8, 11, 19, 35, 26, 42, 0], [0, 2, 27, 10, 25, 43, 36, 30, 44, 47, 21, 49, 37, 40, 0], [0, 6, 4, 17, 3, 9, 34, 22, 0, 14, 7, 48, 33, 41, 50, 20, 29, 38, 0]]
# routes = [[0, 10, 33, 0], [0, 12, 28, 50, 0], [0, 44, 49, 0]]
# routes = [[0, 89, 18, 60, 83, 45, 46, 47, 36, 64, 11, 62, 30, 88, 7, 82, 8, 5, 84, 61, 44, 38, 86, 16, 85, 99, 6, 94, 87, 22, 75, 55, 74, 72, 21, 73, 57, 43, 14, 100, 37, 98, 59, 96, 95, 13, 58, 17, 70, 31, 0], 
        #   [0, 92, 91, 42, 97, 27, 69, 1, 51, 33, 29, 76, 39, 67, 23, 41, 15, 2, 40, 26, 12, 3, 79, 78, 81, 9, 20, 10, 19, 49, 63, 90, 32, 66, 71, 35, 34, 68, 80, 54, 4, 25, 24, 50, 28, 93, 0]]

routes =  [[0, 63, 33, 28, 27, 29, 90, 0, 94, 50, 66, 89, 0], [0, 11, 47, 15, 12, 98, 0, 67, 71, 54, 56, 0, 96, 0, 68, 0], [0, 72, 95, 62, 31, 30, 76, 0, 53, 79, 78, 10, 0, 37, 35, 43, 1, 70, 100, 0], [0, 92, 83, 59, 75, 21, 0, 41, 3, 4, 0, 55, 60, 0], [0, 14, 
82, 52, 73, 16, 87, 57, 0, 9, 97, 17, 13, 74, 77, 0], [0, 42, 5, 45, 2, 69, 0, 23, 20, 24, 48, 25, 58, 0], [0, 39, 36, 38, 44, 61, 0, 88, 7, 8, 46, 6, 0, 34, 26, 32, 91, 80, 0], [0, 65, 64, 19, 0, 86, 22, 49, 84, 93, 0]]
routes_trips = []
for route in routes:
    route= trip_split(route)
    route.append([0,0])
    routes_trips.append(route)

# time,time_slack,time_adjustment_list,demand_slack = find_time_slack_newest(routes_trips,data,initial_data,parameters)

# print('time_slacks',time_slack[1])
# print('time_adjustment_list',time_adjustment_list[1])
# print(data.distances[14,0])
'''
routes = [i for route in routes for i in route if i!=0]
route_dict ={}
for i in routes:
    if i in route_dict.keys():
        print('customer repeated',i)
    else:
        route_dict[i] =1'''

# plot_route(routes,data,initial_data,parameters)
# plot_time(routes,data,initial_data,parameters)

# degree_of_destruction =3

# print(routes)
# print('obj_b4', objective_function_fast(routes,data,initial_data,parameters))

# routes = random_remove_point(routes,data,initial_data,parameters,3)

routes1 = best_insertion_fastest(routes,data,initial_data,parameters)
# routes2 = best_regret2_fastest(routes,data,initial_data,parameters)
# routes = create_trips(routes,data,initial_data,parameters)
# print(data.distances[0,24])
# print(routes1)
# print(routes2)
# assert routes1 == routes2

# routes = initial_greedy_insertion_distance(data,initial_data,parameters)
# print(routes)

print(route_check(routes,data,initial_data,parameters,log=True))
print('obj', objective_function(routes,data,initial_data,parameters))
print('obj_fast', objective_function_fast(routes,data,initial_data,parameters))
# plt.show()

