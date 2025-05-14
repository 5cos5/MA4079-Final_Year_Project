from ALNS import ALNS
from collections import defaultdict

if __name__ == '__main__':
    '''
    ### This is for experiment 1
    FILE_LIST= ['test_data/test1',
                'test_data/test2',
                'test_data/test3',
                'test_data/test4',
                'test_data/test5',
                'test_data/test6']
    
    N_LIST = [6,18,18,18,18,6]
    VEHICLES_LIST = [1,3,1,3,3,1]
    CAPACITY_LIST = [60,60,60,50,60,40]
    '''

    ### This is for experiment 2
    FILE_LIST= ['shuffled_data/C201_shuffled',
            'shuffled_data/C208_shuffled',
            'shuffled_data/R201_shuffled',
            'shuffled_data/R211_shuffled',
            'shuffled_data/RC201_shuffled',
            'shuffled_data/RC208_shuffled']

    N_LIST  = [25]*len(FILE_LIST)
    VEHICLES_LIST = [3]*len(FILE_LIST)
    CAPACITY_LIST = [50]*len(FILE_LIST)

    result_dict = defaultdict(list)
    route_dict = defaultdict(list)
    for i, file in enumerate(FILE_LIST):
        for a in range(5):
            print(f"FILE: {file}, iteration:{a}")
            result, time, best_route = ALNS(file,N_LIST[i],VEHICLES_LIST[i],CAPACITY_LIST[i])
            best_route = tuple(tuple(int(i) for i in route_trips) for route_trips in best_route)
            result_dict[file].append((round(result,5),round(time,3)))
            route_dict[file].append(best_route)

    import json

    with open('result.json', 'w') as fp:
        json.dump(result_dict, fp, sort_keys=True, indent=4)


    with open('route.json', 'w') as fp:
        json.dump(route_dict, fp, sort_keys=True)#, indent=4)  
    print(result_dict)