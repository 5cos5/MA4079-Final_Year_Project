from read_data import Input, InitialData, InputParameters
# from read_data_expt2 import Input, InitialData, InputParameters
from model_check import route_check, objective_function,distance_travelled,goods_delivered,replace_consecutive_depots
# from greedy_insertion import initial_greedy_insertion_distance
from initial import initial_greedy_insertion_distance
from destroy import random_remove_point,greedy_remove_distance,shaw_removal
from repair import best_insertion_fastest,random_cust_best_insertion_fastest,best_regret2_fastest,best_regret2
from trip_generator import create_trips, remove_trips, remove_depots,remove_repeated_depot
# from route import Routes
from plot_data import plot_route, plot_time
import numpy as np
import timeit
from tqdm import tqdm
from math import ceil
import matplotlib.pyplot as plt
import copy
from decimal import Decimal

start = timeit.default_timer()
temperature = 50
cooling= 0.995 # 0<aplha<1
non_improvement = 0
est_cycles = ceil(np.emath.logn(cooling,1e-3/temperature))
early_term_percent = 0.2
early_term = round(est_cycles*early_term_percent)
NO_ITER_PERCENTAGE = 0.005
NO_ITER = round(est_cycles*NO_ITER_PERCENTAGE)
if NO_ITER == 0 :
    NO_ITER = 1
RESET = False
def StoppingCriteria(T,non_improvement):
    return T>1e-3# and non_improvement<early_term

def Accept(obj,temp_obj,temperature):
    rand_no = rng.random()
    # print('Temperature:',temperature)
    # print('Obj_diff', obj - temp_obj)
    # print(np.exp(-(obj - temp_obj)/(temperature)))
    if temp_obj>obj or rand_no < np.exp(-(float(obj) - float(temp_obj))/(temperature)):
        return True
    else:
        return False

N = 25
# FILE = 'shuffled_ordered_data/R211_shuffled_ordered'
# FILE = 'test_data/test6'
# FILE = 'test_data/R208_test'
FILE = 'shuffled_data/R211_shuffled'
# FILE = 'experiment2_data/C201_test'
N_VEHICLES = 3
# CAPACITY = Decimal(100)
CAPACITY = 50
DEMAND_TYPES = 5
VEHICLE_COMPARMENTS = 5
MAX_CAPACITY = np.array([CAPACITY]*DEMAND_TYPES)
EMPTY_CAPACITY = np.array([0]*DEMAND_TYPES)
PRICE = np.array([1]*DEMAND_TYPES)
# ZETA = Decimal('0.01')
ZETA = 0.1
parameters = InputParameters(N_VEHICLES=N_VEHICLES,DEMAND_TYPES=DEMAND_TYPES,VEHICLE_COMPARMENTS=VEHICLE_COMPARMENTS,
                                DEPOT_LOADING_SCALAR= 0.2, MAX_CAPACITY=MAX_CAPACITY,EMPTY_CAPACITY=EMPTY_CAPACITY, ZETA=ZETA, PRICE = PRICE)
data = Input.load_csv(FILE= FILE, N= N, parameters= parameters)

route1_cust_no = [1]
route2_cust_no = [1]
route3_cust_no = [1]
route4_cust_no = [1]

time_start_1 = 0
time_start_2 = 0
time_start_3 = 0
time_start_4 = 0

loading1 = [CAPACITY,CAPACITY,CAPACITY,CAPACITY,CAPACITY]
loading2 = [CAPACITY,CAPACITY,CAPACITY,CAPACITY,CAPACITY]
loading3 = [CAPACITY,CAPACITY,CAPACITY,CAPACITY,CAPACITY]
loading4 = [CAPACITY,CAPACITY,CAPACITY,CAPACITY,CAPACITY]
''''
initial_data = InitialData(
    initial_route = [route1_cust_no, route2_cust_no, route3_cust_no], #route4_cust_no],
    initial_loading= [loading1,loading2,loading3,loading4],
    initial_time= [time_start_1,time_start_2,time_start_3,time_start_4]
)'''

initial_data = InitialData(
    initial_route = [route1_cust_no]*N_VEHICLES, 
    initial_loading= [loading1]*N_VEHICLES,
    initial_time= [time_start_1]*N_VEHICLES
)
animation_list = []
best_animation_list = []

routes = initial_greedy_insertion_distance(data,initial_data,parameters)
print(routes)
animation_list.append(routes)
best_animation_list.append(routes)
print("Objective:", objective_function(routes,data,initial_data,parameters))


def destroy_operator (routes, index, data,initial_data,parameters,degree_of_destruction):
    # print('destroy_index' , destroy_index)
    # print('routes b4 destroy', routes)
    if index == 0: #random_removal
        routes = random_remove_point(routes,data,initial_data,parameters,degree_of_destruction)
    elif index == 1: #worst distance removal
        routes = greedy_remove_distance(routes,data,initial_data,parameters,degree_of_destruction)
    elif index ==2: #shaw removal
        routes = shaw_removal(routes,data,initial_data,parameters,degree_of_destruction)
    # print('routes_b4_replacement', routes)
    routes = replace_consecutive_depots(routes)
    return routes
    

def repair_operator(routes, index, data,initial_data,parameters):
    # print('repair index', repair_index)
    # print('routes b4 repair', routes)
    animation_list.append(routes)
    if index == 0:
        routes = random_cust_best_insertion_fastest(routes,data,initial_data,parameters)
    if index == 1:
        routes1 = best_insertion_fastest(routes,data,initial_data,parameters)
        # routes2 = best_insertion_fast(routes,data,initial_data,parameters)
        # assert routes1 == routes2
        routes = routes1
    elif index == 2:
        # routes1 = best_regret2_fastest(routes,data,initial_data,parameters)
        routes2 = best_regret2(routes,data,initial_data,parameters)
        # print('Route 1', routes1)
        # print('Route 2', routes2)
        # assert routes1 == routes2
        routes = routes2
    # print('routes af repair', routes)
    animation_list.append(routes)
    return routes

def update_weights(weights,prev_score,score,op_used,learning_rate):
    # weights[op_used] = weights[op_used]*(1-learning_rate) + (score-prev_score)*learning_rate
    temp = weights[op_used] + (score-prev_score)*learning_rate
    if temp >0: #if weights would be adjusted to be negative, ignore it
        weights[op_used] += (score-prev_score)*learning_rate
    return weights


def update_weights_new(weights,score,op_used,learning_rate):
    # weights[op_used] = weights[op_used]*(1-learning_rate) + (score-prev_score)*learning_rate
    weights[op_used] = (1-learning_rate)*weights[op_used] + learning_rate*score
    '''
    temp = weights[op_used] + (score-prev_score)*learning_rate
    if temp >0: #if weights would be adjusted to be negative, ignore it
        weights[op_used] += (score-prev_score)*learning_rate'''
    return weights

def update_weights_newest(weights,score,learning_rate,iteration_count):
    # weights[op_used] = weights[op_used]*(1-learning_rate) + (score-prev_score)*learning_rate
    for i in range(len(weights)):
        if iteration_count[i] != 0:
            weights[i] = (1-learning_rate)*weights[i] + learning_rate*(score[i]/iteration_count[i])
        else: # weights not used, just update with reduction
            weights[i] = (1-learning_rate)*weights[i]
    '''
    temp = weights[op_used] + (score-prev_score)*learning_rate
    if temp >0: #if weights would be adjusted to be negative, ignore it
        weights[op_used] += (score-prev_score)*learning_rate'''
    return weights


def normalize(probs):
    prob_factor = 1 / sum(probs)
    return [prob_factor * p for p in probs]

SCORING = [1,0.5,0.25]
# SEED = 69
LEARNING_RATE = 0.1 ## Select a value larger than 0
best_obj = 0
prev_obj = 0
obj = 0
DESTRUCTION_PERCENTAGE = 0.3
DESTROY_OPERTATORS = 3
destroy_weights = [1]*DESTROY_OPERTATORS
REPAIR_OPERTATORS = 3
repair_weights = [1]*REPAIR_OPERTATORS
destroy_score, repair_score = [0]*DESTROY_OPERTATORS, [0]*REPAIR_OPERTATORS
destroy_count, repair_count = [0]*DESTROY_OPERTATORS, [0]*REPAIR_OPERTATORS
curr_obj_list = []
best_obj_list = []
temp_obj_list = []
repair_weights_list= []
destroy_weights_list= []
cycles = 0
rng = np.random.default_rng()
pbar = tqdm(total = est_cycles)
while StoppingCriteria(temperature, non_improvement):
    # print(routes)
    destroy_index = rng.choice([j for j in range(DESTROY_OPERTATORS)],p = normalize(destroy_weights))
    repair_index = rng.choice([j for j in range(REPAIR_OPERTATORS)],p = normalize(repair_weights))
    destroy_weights_list.append(tuple(normalize(destroy_weights)))
    repair_weights_list.append(tuple(normalize(repair_weights)))
    max_destruction = ceil(DESTRUCTION_PERCENTAGE*(len(sum(routes,[])) -sum(routes,[]).count(0)))
    if max_destruction <= 1:
        degree_of_destruction = 1
    else:
        degree_of_destruction = rng.integers(low = 1, high = max_destruction)

    temp_routes = copy.deepcopy(routes)
    temp_routes = destroy_operator(temp_routes,destroy_index,data,initial_data,parameters,degree_of_destruction)
    temp_routes = repair_operator(temp_routes,repair_index,data,initial_data,parameters)
    # temp_routes1 = create_trips(temp_routes,data,initial_data,parameters)

    
    if not route_check(temp_routes,data,initial_data,parameters,log =True): #if route is infeasible skip (this is not supposed to happen)
        print('Repair_operator', repair_index)
        print("INFEASIBLE")
        print('temp_route', temp_routes)
        print('route', routes)
        temp_obj_list.append(0)
        curr_obj_list.append(obj)
        best_obj_list.append(best_obj)
        cycles +=1
        continue
    temp_obj = objective_function(temp_routes,data,initial_data,parameters)
    score = 0

    # print(temp_routes)
    if Accept(obj,temp_obj,temperature) and not temp_routes==routes:
        if temp_obj>obj:
            score = SCORING[1]
        else:
            score = SCORING[2]
        routes = temp_routes
        obj = temp_obj
        non_improvement += 1
        if obj > best_obj:
            non_improvement = 0
            best_obj = obj
            best_route = tuple(tuple(route) for route in routes)
            score = SCORING[0]
            # print("Best_route", best_route)
            print ("Best_obj",best_obj)
            best_animation_list.append(routes)

    else:
        non_improvement +=1


    # if cycles >5: #update weights after first few cycles
    destroy_score[destroy_index] += score
    repair_score[repair_index] += score
    destroy_count[destroy_index] += 1
    repair_count[repair_index] += 1
    if cycles % NO_ITER == 0:
        destroy_weights = update_weights_newest(destroy_weights,destroy_score,LEARNING_RATE,destroy_count)
        repair_weights = update_weights_newest(repair_weights,repair_score,LEARNING_RATE,repair_count)
        destroy_score, repair_score = [0]*DESTROY_OPERTATORS, [0]*REPAIR_OPERTATORS
        destroy_count, repair_count = [0]*DESTROY_OPERTATORS, [0]*REPAIR_OPERTATORS
    # destroy_weights = update_weights_new(destroy_weights,destroy_score,destroy_index,LEARNING_RATE)
    # repair_weights = update_weights_new(repair_weights,repair_score,repair_index,LEARNING_RATE)
    
    temp_obj_list.append(temp_obj)
    curr_obj_list.append(obj)
    best_obj_list.append(best_obj)
    temperature *= cooling
    cycles +=1

    if non_improvement >= early_term: #too many non-imporvement cycles
        # if routes!= best_route: #set candidate route to best route
        if not RESET:
            # print('Reset to best route')
            RESET = True
            routes = list(list(route) for route in best_route)
            non_improvement = 0
        else: #if candidate route same as best route, end cycle early
            print('Too many non-improvements')
            # cycles +=1
            break 
    pbar.update(1)

pbar.close()
print("The difference of time is :",
              timeit.default_timer() - start)

print ("Best obj:", best_obj)
print("Distance Travelled", distance_travelled(best_route, data, initial_data, parameters))
print("Goods Delivered", goods_delivered(best_route, data, initial_data, parameters))
routes= []
for route in best_route:
    routes.append([data.cust_no[i] for i in route])
print ("Best routes:", routes)
print (route_check(best_route,data,initial_data,parameters,log=True))
def flatten_comprehension(matrix):
    return [item for row in matrix for item in row]

cust_visted = flatten_comprehension(best_route)
# print("Customers Visited:", len(set(cust_visted)))
cust_visted = [i for i in cust_visted if i!=0]
print("Customers Visited:", len(cust_visted))
# print(cycles)
# print(len(temp_obj_list))

#plot objective function
fig, ax = plt.subplots()
ax.set_title('Obj Function')
x = range(cycles)
ax.plot(curr_obj_list, c = 'b')
ax.scatter(x,temp_obj_list, c = 'r', s=1)
ax.plot(best_obj_list, c = 'g')
ax.set_ylim(0,float(max(best_obj_list))*1.10)
ax.set_xlim(0,cycles)

# plot destroy weights graph
fig, ax = plt.subplots()
ax.set_title('Destroy Weights')
# make data
y = np.array(destroy_weights_list)
ax.plot(x, y, label = ['RR','BC','SR'])
ax.legend()

# plot repair weights graph
fig, ax = plt.subplots()
ax.set_title('Repair Weights')
# make data
x = range(cycles)
y = np.array(repair_weights_list)
ax.plot(x, y, label = ['RBI','BI','RI'])
ax.legend()

plot_route(best_route,data,initial_data,parameters)
plot_time(best_route,data,initial_data,parameters)
# print("number of iterations to update weights",NO_ITER)
plt.show()
# print(animation_list)
'''
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

with open('animation.json', 'w') as fp: 
    json.dump(animation_list, fp, cls=NpEncoder)

with open('best_animation.json', 'w') as fp: 
    json.dump(best_animation_list, fp, cls=NpEncoder)'''
