import numpy as np
from geometry import dist_from_point_to_polyline, dense_polyline


def get_front_vehicle(lane_path, vehicle_list, ego_vehicle):
    """
    input:
    lane_path: np.array
    vehicle_list: list ([x1,y1],[x2,y2],....)
    
    return object list
    """
    
    ego_x = ego_vehicle[0]
    ego_y = ego_vehicle[1]
    
    ego2lane, ego2lane_head, ego2lane_tail = dist_from_point_to_polyline(ego_x, ego_y, lane_path)
    
    # print(ego2lane, ego2lane_head, ego2lane_tail)
    v2lane_front = []
    
    for v_id, vehicle in enumerate(vehicle_list):
        vehicle_x = vehicle[0]
        vehicle_y = vehicle[1]
        d2lane, d2lane_head, d2lane_tail = dist_from_point_to_polyline(vehicle_x, vehicle_y, lane_path)
        if d2lane_head > ego2lane_head:
            v2lane_front.append([v_id, d2lane, d2lane_head, d2lane_tail])
        
    v2lane_front = np.array(v2lane_front)
    
    front_vehicle_id = v2lane_front[np.argmin(v2lane_front[:, 2])][0]
    
    print("front vehicle id is:", front_vehicle_id)
    

if __name__ == '__main__':
    
    lane_path_ori = np.array([[0,0],[20,0],[20,50]])
    lane_path = dense_polyline(lane_path_ori,5)

    vehicle_list = [[0,0],[10,0],[20,0]]
    ego_vehicle = [5,1]
    
    get_front_vehicle(lane_path, vehicle_list, ego_vehicle)