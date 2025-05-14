import pyomo.environ as pyo
from scipy.spatial.distance import pdist, squareform
import numpy as np
import pandas as pd

def MILP(FILE,N,N_VEHICLES,CAPACITY,R):
    FILE_NAME = "Data/"+FILE+".csv"
    data = pd.read_csv(FILE_NAME)
    data.columns = data.columns.astype(str)

    PLANNING_TIME = 0
    CUSTOMERS_LIST = []
    data = data.loc[(data['DUE DATE'] >= PLANNING_TIME) | (data['CUST NO.'].isin(CUSTOMERS_LIST))]

    DEMAND_TYPES = 5
    VEHICLE_COMPARMENTS = 5

    vehicle_capacities = np.array([[CAPACITY]*DEMAND_TYPES]*N_VEHICLES)
    df = data[:N+1]
    cust_no = np.array(df.loc[:,'CUST NO.'])
    demands = np.array(df.loc[:,'DEMAND'])
    service_time = np.array(df.loc[:,'SERVICE TIME'])
    demands_zeros = np.zeros((len(demands), DEMAND_TYPES-1))
    demands = np.hstack((demands.reshape(N+1,1),demands_zeros))
    coordinates = np.array(df.loc[:,['XCOORD','YCOORD']])
    start_time_windows = np.array(df.loc[:,'READY TIME'])
    end_time_windows = np.array(df.loc[:,'DUE DATE']) 

    demands_index = [i for i in range(1,N+1)]

    START_TIME = start_time_windows[0]
    END_TIME = end_time_windows[0]

    demands_shuffled = np.array(df.iloc[:,4:9])

    distances = squareform(pdist(coordinates, metric="euclidean"))
    def trunc(values, decs=1):
        return np.trunc(values*10**decs)/(10**decs)
    distances = trunc(distances)
    travel_time = distances
    model = pyo.ConcreteModel()

    # ## Sets
    model.D = pyo.Set(initialize=demands_index) #set of vessels
    model.N = pyo.Set(initialize=range(len(demands))) # set of vessels+depot
    model.N1 = pyo.Set(initialize=range(len(demands)+1)) # set of vessels+depot+end
    model.A = pyo.Set(initialize=[(i, j) for i in model.N for j in model.N if i != j]) #set of arcs
    model.K = pyo.Set(initialize=range(N_VEHICLES)) #set of vehicles
    model.R  = pyo.Set(initialize=range(R)) # set of trips
    model.F = pyo.Set(initialize = range(DEMAND_TYPES)) #set of demand types


    zeta = 0.1 # travel cost per distance factor
    price = [10]*DEMAND_TYPES

    # ## Parameters
    model.q = pyo.Param(model.K, model.F, initialize={(i,f): vehicle_capacities[i,f] for i in model.K for f in model.F}) #capacity of vehicles
    model.c = pyo.Param(model.A, initialize={(i, j): zeta*distances[i, j] for (i, j) in model.A}) #cost of travel per arc
    model.n = pyo.Param(model.N, model.F, initialize={(i,f): demands_shuffled[i,f] for i in model.N for f in model.F}) #loading of all nodes
    model.r = pyo.Param(model.N, model.F, initialize={(i,f): price[f]*demands_shuffled[i,f] for i in model.N for f in model.F}) #revenue of all nodes
    model.s = pyo.Param(model.N, initialize=service_time) #service time per node
    model.t = pyo.Param(model.N, model.N, initialize={(i, j): travel_time[i, j] for i in model.N for j in model.N}) #travel time per arc)
    model.a = pyo.Param(model.N, initialize=start_time_windows) #start time windows per node
    model.b = pyo.Param(model.N, initialize=end_time_windows) #end time windows per node


    # ## Variables
    new_start_time_windows = np.append(start_time_windows, START_TIME)
    new_start_time_windows = [[new_start_time_windows]*R]*N_VEHICLES

    new_end_time_windows = np.append(end_time_windows, END_TIME)
    new_end_time_windows = [[new_end_time_windows]*R]*N_VEHICLES
    def time_windows(model, i,k,r):
        return (new_start_time_windows[k][r][i], new_end_time_windows[k][r][i])

    model.X = pyo.Var(model.A, model.K, model.R, within=pyo.Binary) # decision to move along arc ij by vehicle K on trip R
    model.Y = pyo.Var(model.N, model.K, model.R, within=pyo.Binary) # decision of if node N is visited by vehicle K on trip R
    model.T = pyo.Var(model.N1, model.K, model.R, within=pyo.NonNegativeReals, bounds= time_windows) # visit time at node N by vehicle K on trip R
    model.S = pyo.Var(model.K, model.R, within=pyo.NonNegativeReals, bounds= (0,CAPACITY*DEMAND_TYPES)) #serivce time at the depot for the start of each trip
    model.L = pyo.Var([0,N+1], model.K, model.R, model.F, within=pyo.NonNegativeReals, bounds= (0,CAPACITY)) # fuel load on vehicle K on trip R for fuel type F
    model.U = pyo.Var(model.N, model.K, model.R, model.F, within=pyo.NonNegativeReals, bounds= (0,CAPACITY))

    M = np.zeros((N+1,N+1))
    beta = 0.2 # rate of fuel transfer at the depot
    for i in model.N:
        for j in model.N:
            M[i,j] = model.b[i] + model.s[i] + model.t[i,j] - model.a[j]
    for j in model.N:
        M[0,j] = model.b[0] + CAPACITY*DEMAND_TYPES*beta + model.t[0,j] - model.a[j]

    # ## Constraints
    def visit_once_rule(model,i): #2/13
        return sum(model.Y[i,k,r] for k in model.K for r in model.R) == 1
    model.visit_once_rule = pyo.Constraint(model.D, rule=visit_once_rule)

    def arcs_out_rule(model, i, k,r): #3/14 modifyied Arcs out rule
        return sum(model.X[i, j, k,r] for j in model.N if i != j) == model.Y[i,k,r]
    model.arcs_out_rule = pyo.Constraint(model.N, model.K, model.R, rule=arcs_out_rule)

    def arcs_in_rule(model, i, k,r): #3/14 modifyied Arcs in rule
        return sum(model.X[j, i, k,r] for j in model.N if i != j) == model.Y[i,k,r]
    model.arcs_in_rule = pyo.Constraint(model.N, model.K, model.R, rule=arcs_in_rule)

    def feasbile_time_along_arc(model,i,j, k,r): #71
        if i == j:
            return pyo.Constraint.Skip
        else:
            # return model.T[i,k,r] + model.s[i] + model.t[i,j] <= model.T[j,k,r] + M*(1-model.X[i,j,k,r])
            return model.T[i,k,r] + model.s[i] + model.t[i,j] <= model.T[j,k,r] + M[i,j]*(1-model.X[i,j,k,r])
    model.feasbile_time_along_arc = pyo.Constraint(model.D, model.D, model.K, model.R, rule=feasbile_time_along_arc)

    def feasbile_start_time (model,j,k,r): #72 
        # return model.T[0,k,r] + model.S[k,r] + model.t[0,j] <= model.T[j,k,r] + M*(1-model.X[0,j,k,r])
        return model.T[0,k,r] + model.S[k,r] + model.t[0,j] <= model.T[j,k,r] + M[0,j]*(1-model.X[0,j,k,r])
    model.feasbile_start_time = pyo.Constraint(model.D, model.K, model.R, rule=feasbile_start_time)

    def feasbile_end_time (model,i,k,r): #66
        # return model.T[i,k,r] + model.s[i] + model.t[i,0] <= model.T[N+1,k,r] + M*(1-model.X[i,0,k,r])
        return model.T[i,k,r] + model.s[i] + model.t[i,0] <= model.T[N+1,k,r] + M[i,0]*(1-model.X[i,0,k,r])
    model.feasbile_end_time = pyo.Constraint(model.D, model.K, model.R, rule=feasbile_end_time)

    def trip_end_before_start (model, k,r): #67
        if r == model.R.last():
            return pyo.Constraint.Skip
        else:
            return model.T[N+1,k,r] <= model.T[0,k,r+1]
    model.trip_end_before_start = pyo.Constraint(model.K, model.R, rule=trip_end_before_start)    

    def trip_end_after_start (model, k,r): #modified (necessary)
        return model.T[0,k,r] + model.S[k,r] <= model.T[N+1,k,r]
    model.trip_end_after_start = pyo.Constraint(model.K, model.R, rule=trip_end_after_start)    

    # ## Service time at depot
    full_capacity = [CAPACITY]*VEHICLE_COMPARMENTS

    def service_time_at_depot (model,k,r): #73 modified
        if r == model.R.first():
            return model.S[k,r] == 0 #beta * (sum(full_capacity)- sum(model.L_initial[k,f] for f in model.F))
        else:
            return model.S[k,r] == beta* (sum(full_capacity) - sum(model.L[N+1,k,r-1,f] for f in model.F))
    model.service_time_at_depot = pyo.Constraint(model.K, model.R, rule=service_time_at_depot)



    # ## Vehicle Loading
    def fuel_del_to_visit(model,i,k,r,f): #if cust i is visited, U>0 else set  U = 0
        return model.U[i,k,r,f] == model.Y[i,k,r] * model.n[i,f]

        #  return model.U[i,k,r,f] <= model.Y[i,k,r] * CAPACITY
    model.fuel_del_to_visit = pyo.Constraint(model.N, model.K, model.R, model.F,rule=fuel_del_to_visit)

    def fuel_load_in_veh_at_start(model,k,r,f): #10
        if r == model.R.first():
            return pyo.Constraint.Skip #loading for first trip would be handled later
        else:
            # return pyo.Constraint.Skip
            return model.L[0,k,r,f] == full_capacity[f]
    model.fuel_load_in_veh_at_start = pyo.Constraint(model.K,model.R, model.F,rule=fuel_load_in_veh_at_start)

    def load_in_veh_at_end(model,k,r,f): #fuel load at end same as or exceeds the load for the trip
        return model.L[N+1,k,r,f] == model.L[0,k,r,f] - sum(model.U[i,k,r,f] for i in model.N) 
    model.load_in_veh_at_end = pyo.Constraint(model.K,model.R, model.F,rule=load_in_veh_at_end)



    #obj with travel costs
    model.obj = pyo.Objective(
        expr=sum(
            model.Y[i, k,r] * np.abs(model.r[i,f])
            for i in model.D
            for k in model.K
            for r in model.R
            for f in model.F
        ) - sum(model.X[i, j, k,r] * model.c[i,j]
            for (i, j) in model.A
            for k in model.K
            for r in model.R),
        sense=pyo.maximize,
    )


    # ## Inital Conditions
    full_capacity = [CAPACITY]*VEHICLE_COMPARMENTS
    empty_capacity = [0]*VEHICLE_COMPARMENTS

    route1_cust_no = [1]
    route2_cust_no = [1]
    route3_cust_no = [1]

    depot_index = np.where(cust_no == 1)[0][0]
    if len(route1_cust_no) != 0:
        route1_index = np.where(cust_no == route1_cust_no)[0][0]
    else:
        route1_index = None
    if len(route2_cust_no) != 0:
        route2_index = np.where(cust_no == route2_cust_no)[0][0]
    else:
        route2_index = None
    if len(route3_cust_no) != 0:
        route3_index = np.where(cust_no == route3_cust_no)[0][0]
    else:
        route3_index = None

    #time start at depot
    time_start_1 = 0
    time_start_2 = 0
    time_start_3 = 0

    #loading of vehicle before top up
    loading1 = full_capacity
    loading2 = full_capacity
    loading3 = full_capacity

    #check that given load does not exceed vehicle capacity
    for i, cap in enumerate(full_capacity):
        if (loading1[i] > cap) or (loading2[i] > cap) or (loading3[i] > cap) :
            raise Exception('Capacity over limit')
        

    routes_initial = [route1_cust_no, route2_cust_no, route3_cust_no]
    time_start_initial = [time_start_1,time_start_2,time_start_3]
    loading_initial = [loading1,loading2,loading3]


    def inital_loading(model,k,f):
        return model.L[0,k,0,f] == loading_initial[k][f]

    model.inital_loading = pyo.Constraint(model.K,model.F,rule=inital_loading)

    if route1_index != 0:
        def inital_route_1(model):
            return model.X[0,route1_index,0,0] == 1
        model.inital_route_1 = pyo.Constraint(rule=inital_route_1)
    if route2_index != 0:
        def inital_route_2(model):
            return model.X[0,route2_index,1,0] == 1
        model.inital_route_2 = pyo.Constraint(rule=inital_route_2)
    if route3_index != 0:
        def inital_route_3(model):
            return model.X[0,route3_index,2,0] == 1
        model.inital_route_3 = pyo.Constraint(rule=inital_route_3)

    def initial_time_start_at_depot (model,k):
        return model.T[0,k,0] == time_start_initial[k]
    model.initial_time_start_at_depot = pyo.Constraint(model.K, rule=initial_time_start_at_depot)


    ## Reducing Arcs
    for arc in model.A:
        i,j = arc[0],arc[1]
        if start_time_windows[i] +service_time[i] +travel_time[i,j] > end_time_windows[j]:
            for k in model.K:
                for r in model.R:
                    model.X[i,j,k,r].fix(0) #remove these arcs




    TIME_LIMIT = 180
    THREADS = 4
    MIPFocus = 0 #gurobi, default is 0, set to 1 to focus more on finding feasible solutions

    SOVLER_ENGINE = 'gurobi'
    #solvers glpk  appsi_highs cplex gurobi gurobi_persistent

    solver = pyo.SolverFactory(SOVLER_ENGINE)
    solver.options['timelimit'] = TIME_LIMIT
    solver.options['threads'] = THREADS
    solver.options['MIPFocus'] = MIPFocus
    temp = FILE.split('/')
    logfile = temp[1]+str(N)+'.txt'
    sol = solver.solve(model, tee= True, logfile= logfile) #, warmstart=True , logfile= 'log.txt'


    print(sol.solver.status)
    print(sol.solver.termination_condition)
    print (model.obj())

    print ('Total amount of goods delivered:',sum(
            model.X[i, j, k,r].value * np.abs(model.n[i,f])
            for (i, j) in model.A
            for k in model.K
            for r in model.R
            for f in model.F
            ) )

    print ('Total number of vessels visited:', sum(model.Y[i,k,r].value
                                                for i in model.D
                                                for k in model.K
                                                for r in model.R))

    print ('Total travel time:', sum(model.X[i,j,k,r].value * model.t[i,j]
                                                for (i,j) in model.A
                                                for k in model.K
                                                for r in model.R))


FILE_LIST= ['experiment3_data/C208_test',
            'experiment3_data/C208_test',
            'experiment3_data/R201_test',
            'experiment3_data/R211_test',
            'experiment3_data/RC201_test',
            'experiment3_data/RC208_test']

N_LIST  = [25]*len(FILE_LIST)
VEHICLES_LIST = [2]*len(FILE_LIST)
CAPACITY_LIST = [100]*len(FILE_LIST)
R_LIST = [4]*len(FILE_LIST)



for i, file in enumerate(FILE_LIST):
    MILP(file,N_LIST[i], VEHICLES_LIST[i], CAPACITY_LIST[i], R_LIST[i])
    
