from ALNS import ALNS
from collections import defaultdict

if __name__ == '__main__':
    FILE_LIST= ['experiment3_data/C201_test',
                'experiment3_data/C208_test',
                'experiment3_data/R201_test',
                'experiment3_data/R211_test',
                'experiment3_data/RC201_test',
                'experiment3_data/RC208_test']
    
    VEHICLES_LIST = [2]*len(FILE_LIST)
    N_LIST  = [25]*len(FILE_LIST)
    CAPACITY_LIST = [100,100,100,100,100,100]

    result_dict = defaultdict(list)
    route_dict = defaultdict(list)

    for i, file in enumerate(FILE_LIST):
        for a in range(5):
            print(f"FILE: {file}, iteration:{a+1}")
            result, time,visited, best_route = ALNS(file, N_LIST[i], VEHICLES_LIST[i], CAPACITY_LIST[i])
            best_route = tuple(tuple(int(i) for i in route_trips) for route_trips in best_route)
            result_dict[file].append((float(result),round(time,3),visited))
            route_dict[file].append(best_route)
            # if result < 1509.5:
            #     break

    import json

    with open('result.json', 'w') as fp:
        json.dump(result_dict, fp, sort_keys=True, indent=4)

    with open('route.json', 'w') as fp:
        json.dump(route_dict, fp, sort_keys=True)#, indent=4)  
    print(result_dict)