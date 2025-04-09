from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
# from numba import jit

@dataclass
class InitialData():
    initial_route: list
    initial_loading: list
    initial_time: list

@dataclass
class InputParameters():
    N_VEHICLES :int
    DEMAND_TYPES: int
    VEHICLE_COMPARMENTS: int
    DEPOT_LOADING_SCALAR: int
    MAX_CAPACITY: np.array([0])
    EMPTY_CAPACITY: np.array([0])
    ZETA: int
    PRICE: np.array([0])


@dataclass
class Input():
    cust_no: np.array([0])
    indexes: np.array([0])
    coordinates: np.array([0])
    distances: np.array([0])
    service_time: np.array([0])
    demands: np.array([0])
    start_time_windows: np.array([0])
    end_time_windows: np.array([0])
    total_demand: np.array([0])
    revenue: np.array([0])
    total_revenue: np.array([0])

    @classmethod
    def load_csv(cls, FILE:str, N:int, parameters):
        FILE_NAME = "C:/Users/caleb/Desktop/Study/FYP/Data/"+FILE+".csv"
        data = pd.read_csv(FILE_NAME)
        data.columns = data.columns.astype(str)
        df = data[:N+1]
        df = df.loc[(df['DEMAND'] <= parameters.MAX_CAPACITY[0])]
        cust_no= np.array(df.loc[:,'CUST NO.'])
        indexes = np.array([i for i in range(len(cust_no))])
        total_demand = np.array(df.loc[:,'DEMAND'])
        service_time = np.array(df.loc[:,'SERVICE TIME'])#total_demand
        coordinates = np.array(df.loc[:,['XCOORD','YCOORD']])
        start_time_windows = np.array(df.loc[:,'READY TIME'])
        end_time_windows = np.array(df.loc[:,'DUE DATE'])

        demands = np.array(df.iloc[:,4:9])
        revenue = demands *parameters.PRICE
        total_revenue = np.sum(revenue, axis = 1)
        distances = squareform(pdist(coordinates, metric="euclidean"))
        def trunc(values, decs=1):
            return np.trunc(values*10**decs)/(10**decs)
        distances = trunc(distances)
        # distances = np.round(distances, decimals=2)

        return cls (indexes = indexes, cust_no=cust_no,coordinates = coordinates, distances = distances, service_time= service_time, 
                    demands = demands,start_time_windows = start_time_windows, end_time_windows = end_time_windows,total_demand = total_demand,
                    revenue = revenue, total_revenue = total_revenue)
    


if __name__ == '__main__':
    N = 30
    FILE = 'R201'
    data = Input.load_csv(FILE= FILE, N= N)
    print(data.cust_no)
    
    quit()