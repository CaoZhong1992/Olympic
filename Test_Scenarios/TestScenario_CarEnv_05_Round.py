import glob
import sys

try:    	
    sys.path.append(glob.glob('/home/zhcao/Downloads/CARLA_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.8-linux-x86_64.egg')[0])
    sys.path.append(glob.glob('/home/zhcao/Downloads/CARLA_0.9.13/PythonAPI/carla')[0])

    # sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
	# 	sys.version_info.major,
	# 	sys.version_info.minor,
	# 	'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
	pass

import carla
import time
import numpy as np
import math
import random
from random import randint
from carla import Location, Rotation, Transform, Vector3D, VehicleControl
from gym import core, error, spaces, utils
from gym.utils import seeding
from agents.navigation.global_route_planner import GlobalRoutePlanner
# from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

from Agent.zzz.dynamic_map import Lanepoint, Lane, Vehicle
from Agent.zzz.tools import *
from geometry import dist_from_point_to_polyline, dense_polyline
from Planning_library.coordinates import Coordinates

global x_max
x_max = 90
global x_min
x_min = -90
global y_max
y_max = 60
global y_min
y_min = -120
global circle_center
circle_center = carla.Location(-10, -43, 0)
global stopped_time
stopped_time = np.zeros(1000000)
OBSTACLES_CONSIDERED = 3

global start_point
start_point = Transform()
start_point.location.x = 32
start_point.location.y = -43
start_point.location.z = 1
start_point.rotation.pitch = 0
start_point.rotation.yaw = -90
start_point.rotation.roll = 0

global goal_point
goal_point = Transform()
goal_point.location.x = -51
goal_point.location.y = -44
goal_point.location.z = 1
goal_point.rotation.pitch = 0
goal_point.rotation.yaw = 0 
goal_point.rotation.roll = 0

class CarEnv_05_Round:

    def __init__(self):
        
        # CARLA settings
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        if self.world.get_map().name != 'Carla/Maps/Town05':
            self.world = self.client.load_world('Town05')
        self.world.set_weather(carla.WeatherParameters(cloudiness=50, precipitation=10.0, sun_altitude_angle=30.0))
        settings = self.world.get_settings()
        settings.no_rendering_mode = False
        self.dt = 0.1
        settings.fixed_delta_seconds = self.dt # Warning: When change simulator, the delta_t in controller should also be change.
        settings.substepping = True
        settings.max_substep_delta_time = 0.02  # fixed_delta_seconds <= max_substep_delta_time * max_substeps
        settings.max_substeps = 10
        settings.synchronous_mode = True
        self.world.apply_settings(settings)
        self.free_traffic_lights(self.world)

        self.tm = self.client.get_trafficmanager(8000)
        # self.tm.set_hybrid_physics_mode(True)
        # self.tm.set_hybrid_physics_radius(70.0)
        self.tm.set_random_device_seed(0)

        actors = self.world.get_actors().filter('vehicle*')
        for actor in actors:
            actor.destroy()

        # Generate Reference Path
        self.global_routing()
        global start_point
        global goal_point
        self.start_point = start_point
        self.goal_point = goal_point

        # RL settingss
        self.action_space = spaces.Discrete(3)
        self.low  = np.array([0,  0, 0, 0, 0,0,  0, 0, 0, 0], dtype=np.float64)
        self.high = np.array([1,  1, 1, 1, 1,1,  1, 1, 1, 1], dtype=np.float64)    
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float64)
        self.state_dimension = 20
        
        self.state_vehicle = [] # save the vehicle in states


        # Ego Vehicle Setting
        self.ego_vehicle_bp = random.choice(self.world.get_blueprint_library().filter('vehicle.lincoln.mkz_2020'))
        if self.ego_vehicle_bp.has_attribute('color'):
            color = '255,255,0'
            self.ego_vehicle_bp.set_attribute('color', color)
            self.ego_vehicle_bp.set_attribute('role_name', "hero")
        self.ego_collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.ego_vehicle = None
        self.stuck_time = 0
        
        # Env Vehicle Setting
        self.env_vehicle_bp = random.choice(self.world.get_blueprint_library().filter('vehicle.audi.tt'))
        if self.env_vehicle_bp.has_attribute('color'):
            color = '0,0,255'
            self.env_vehicle_bp.set_attribute('color', color)
        if self.env_vehicle_bp.has_attribute('driver_id'):
            driver_id = random.choice(self.env_vehicle_bp.get_attribute('driver_id').recommended_values)
            self.env_vehicle_bp.set_attribute('driver_id', driver_id)
            self.env_vehicle_bp.set_attribute('role_name', 'autopilot')
        
        # Debug setting
        self.debug = self.world.debug
        self.should_debug = True

        # Control Env Vehicle
        # self.has_set = np.zeros(1000000)
        # self.stopped_time = np.zeros(1000000)   

        # Record
        self.log_dir = "record.txt"
        self.task_num = 0
        self.stuck_num = 0
        self.collision_num = 0

        # Case
        self.case_id = 0
       
    def free_traffic_lights(self, carla_world):
        traffic_lights = carla_world.get_actors().filter('*traffic_light*')
        for tl in traffic_lights:
            tl.set_green_time(10)
            tl.set_red_time(0)

    def global_routing(self):
        start = start_point
        goal = goal_point
        print("Calculating Global Route")
        
        # first route
        grp = GlobalRoutePlanner(self.world.get_map(), sampling_resolution=1) # Carla 0913
        current_route = grp.trace_route(carla.Location(start.location.x,
                                                start.location.y + 20,
                                                start.location.z),
                                carla.Location(goal.location.x,
                                                goal.location.y+20,
                                                goal.location.z))
        t_array = []
        self.ref_path = Lane()
        for wp in current_route:
            lanepoint = Lanepoint()
            lanepoint.position.x = wp[0].transform.location.x 
            lanepoint.position.y = wp[0].transform.location.y
            self.ref_path.central_path.append(lanepoint)
            t_array.append(lanepoint)
        self.ref_path.central_path_array = np.array(t_array)
        self.ref_path.speed_limit = 60/3.6 # m/s

        ref_path_ori = convert_path_to_ndarray(self.ref_path.central_path)
        self.ref_path_array = dense_polyline2d(ref_path_ori, 2)
        self.ref_path_tangets = np.zeros(len(self.ref_path_array))
        self.real_time_ref_path_array = self.ref_path_array
        # second route
        grp2 = GlobalRoutePlanner(self.world.get_map(), sampling_resolution=1) # Carla 0913
        current_route2 = grp2.trace_route(carla.Location(goal.location.x,
                                                goal.location.y - 20,
                                                goal.location.z),
                                carla.Location(start.location.x,
                                                start.location.y -20,
                                                start.location.z))
        t_array = []
        self.ref_path2 = Lane()
        for wp in current_route2:
            lanepoint = Lanepoint()
            lanepoint.position.x = wp[0].transform.location.x 
            lanepoint.position.y = wp[0].transform.location.y
            self.ref_path2.central_path.append(lanepoint)
            t_array.append(lanepoint)
        self.ref_path2.central_path_array = np.array(t_array)
        self.ref_path2.speed_limit = 60/3.6 # m/s

        ref_path_ori2 = convert_path_to_ndarray(self.ref_path2.central_path)
        self.ref_path_array2 = dense_polyline2d(ref_path_ori2, 2)
        self.ref_path_tangets2 = np.zeros(len(self.ref_path_array2))

    def ego_vehicle_stuck(self, stay_thres = 10):    
        ego_vehicle_velocity = math.sqrt(self.ego_vehicle.get_velocity().x ** 2 + self.ego_vehicle.get_velocity().y ** 2 + self.ego_vehicle.get_velocity().z ** 2)
        if ego_vehicle_velocity >= 0.05:
            self.stuck_time = time.time()

        if time.time() - self.stuck_time > stay_thres:
            return True
        
        return False

    def ego_vehicle_pass(self):
        global goal_point
        ego_location = self.ego_vehicle.get_location()
        if ego_location.distance(goal_point.location) < 10:
            return True
        else:
            return False

    def ego_vehicle_collision(self, event):
        self.ego_vehicle_collision_sign = True

    def wrap_state(self, use_ego_coordinate = True):
        # state = [0 for i in range((OBSTACLES_CONSIDERED + 1) * 4)]
        state_ori  = []
        attention_vehicles = []

        ego_vehicle_state = [self.ego_vehicle.get_location().x,
                             self.ego_vehicle.get_location().y,
                             self.ego_vehicle.get_velocity().x,
                             self.ego_vehicle.get_velocity().y,
                             self.ego_vehicle.get_transform().rotation.yaw / 180.0 * math.pi]

        state_ori.append(ego_vehicle_state)
        

        actor_list = self.world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        front_vehicle = self.get_front_vehicle(self.real_time_ref_path_array, vehicle_list)
        closest_vehicle_list = self.get_closest_vehicle(vehicle_list)
        
        if front_vehicle is not None:
            attention_vehicles.append(front_vehicle)
        
        for vehicle in closest_vehicle_list:
            if front_vehicle is not None and vehicle.id == front_vehicle.id:
                continue
            attention_vehicles.append(vehicle)
            
        for vehicle in attention_vehicles:
            vehicle_state = [vehicle.get_location().x,
                             vehicle.get_location().y,
                             vehicle.get_velocity().x,
                             vehicle.get_velocity().y,
                             vehicle.get_transform().rotation.yaw / 180.0 * math.pi]
            
            state_ori.append(vehicle_state)
        
        self.attention_vehicles = attention_vehicles
        
        if self.should_debug:
            for point in self.real_time_ref_path_array:
                self.debug.draw_point(carla.Location(x=point[0],y=point[1],z=0),size=0.05,color=carla.Color(r=255,g=255,b=255),life_time = 0.5)
            
            if front_vehicle is not None:
                fv_loc = front_vehicle.get_location()
                self.debug.draw_point(carla.Location(x=fv_loc.x,y=fv_loc.y,z=fv_loc.z+1), life_time=0.5)
                

        if use_ego_coordinate:
            state = []
            for v_state in state_ori:
                ego_vehicle_coordiate = Coordinates(ego_vehicle_state[0],ego_vehicle_state[1],ego_vehicle_state[4])                
                state.append(list(ego_vehicle_coordiate.transfer_coordinate(v_state[0],v_state[1],
                                                                         v_state[2],v_state[3],
                                                                         v_state[4])))
        else:
            state = state_ori

        return np.array(state).flatten(), np.array(state_ori).flatten()

    def draw_attenton(self, ego_attention):
    
        sum_ego_attention = np.sum(ego_attention[0], axis=1)
        norm_ego_attention = sum_ego_attention/max(abs(sum_ego_attention))
        
        if self.should_debug:
            for v_i, vehicle in enumerate(self.attention_vehicles):
                if not vehicle.is_alive:
                    continue
                thickness = float(max(0.2, abs(norm_ego_attention[v_i+1])))
                if norm_ego_attention[v_i+1] < 0:
                    color = carla.Color(r=211, g=211, b=211, a=255)
                else:
                    color = carla.Color(r=255, g=0, b=0, a=255)
                
                self.debug.draw_line(vehicle.get_location(),
                                    self.ego_vehicle.get_location(), thickness=thickness, color=color, life_time=0.5)

    def get_front_vehicle(self, lane_path, vehicle_list, d_thres=1):
        """
        input:
        lane_path: np.array
        vehicle_list: list ([x1,y1],[x2,y2],....)
        
        return object list
        """
       
        ego_x = self.ego_vehicle.get_location().x
        ego_y = self.ego_vehicle.get_location().y
        
        ego2lane, ego2lane_head, ego2lane_tail = dist_from_point_to_polyline(ego_x, ego_y, lane_path)
        
        # print(ego2lane, ego2lane_head, ego2lane_tail)
        v2lane_front = []
        for v_id, vehicle in enumerate(vehicle_list):
            if vehicle.attributes['role_name'] == "hero":
                continue
            vehicle_x = vehicle.get_location().x
            vehicle_y = vehicle.get_location().y
            d2lane, d2lane_head, d2lane_tail = dist_from_point_to_polyline(vehicle_x, vehicle_y, lane_path)
            if abs(d2lane) > d_thres:
                continue
            if d2lane_head > ego2lane_head:
                v2lane_front.append([v_id, d2lane, d2lane_head, d2lane_tail, vehicle])

        v2lane_front = np.array(v2lane_front)
        if len(v2lane_front) > 0:
            # front_vehicle_id = v2lane_front[np.argmin(v2lane_front[:, 2])][0]
            front_vehicle = v2lane_front[np.argmin(v2lane_front[:, 2])][4]
            # print("front vehicle id is:", front_vehicle_id)
            return front_vehicle
        else:
            return None

    def get_closest_vehicle(self, vehicle_list, d_thres=50):
        
        d_list = []
        
        for v_id, vehicle in enumerate(vehicle_list):
            if vehicle.attributes['role_name'] == "hero":
                continue
            
            d = vehicle.get_location().distance(self.ego_vehicle.get_location())
            
            if d>d_thres:
                continue
            
            d_list.append([v_id, d])
        

        closest_vehicle_list = []
        d_list = np.array(d_list)
          
        while len(closest_vehicle_list)<3:
            if len(d_list) == 0:
                return closest_vehicle_list
            close_id = np.argmin(d_list[:,1])
            closest_vehicle_list.append(vehicle_list[int(d_list[close_id][0])])
            d_list = np.delete(d_list, close_id, 0)

        return closest_vehicle_list
                                            
    def record_information_txt(self):
        if self.task_num > 0:
            stuck_rate = float(self.stuck_num) / float(self.task_num)
            collision_rate = float(self.collision_num) / float(self.task_num)
            pass_rate = 1 - ((float(self.collision_num) + float(self.stuck_num)) / float(self.task_num))
            fw = open(self.log_dir, 'a')   
            # Write num
            fw.write(str(self.task_num)) 
            fw.write(", ")
            fw.write(str(self.case_id)) 
            fw.write(", ")
            fw.write(str(self.stuck_num)) 
            fw.write(", ")
            fw.write(str(self.collision_num)) 
            fw.write(", ")
            fw.write(str(stuck_rate)) 
            fw.write(", ")
            fw.write(str(collision_rate)) 
            fw.write(", ")
            fw.write(str(pass_rate)) 
            fw.write("\n")
            fw.close()               
            print("[CARLA]: Record To Txt: All", self.task_num, self.stuck_num, self.collision_num, self.case_id )

    def clean_task_nums(self):
        self.task_num = 0
        self.stuck_num = 0
        self.collision_num = 0

    def reset(self):    

        # Ego vehicle
        while not self.spawn_ego_veh():
            self.world.tick()
        
        self.world.tick()
        # State
        state, state_ori = self.wrap_state()

        # Record
        self.record_information_txt()
        self.task_num += 1
        self.case_id += 1

        return state, state_ori
    
    def step(self, action, **kw):
        # Control ego vehicle
        throttle = max(0,float(action[0]))  # range [0,1]
        brake = max(0,-float(action[0])) # range [0,1]
        steer = action[1] # range [-1,1]
        self.ego_vehicle.apply_control(carla.VehicleControl(throttle = throttle, brake = brake, steer = steer))
        # self.spawn_random_veh()
        # self.draw_attenton(kw['ego_attention'])
        print("ego_attention:", kw['ego_attention'])
        
        self.world.tick()

        # State
        state, state_ori = self.wrap_state()

        reward = self.reward_function(state)
        
        # If finish
        done = False
        if self.ego_vehicle_collision_sign:
            self.collision_num += + 1
            done = True
            reward = - 10
            print("[CARLA]: Collision!")
        
        # if self.ego_vehicle_pass():
        #     done = True
        
        if self.ego_vehicle_stuck():
            self.stuck_num += 1
            done = True
            reward = 0.0
            print("[CARLA]: Stuck!")

        return state, reward, done, state_ori

    def reward_function(self, state):
        v = math.sqrt(state[2]**2 + state[3]**2)
        reward = math.exp(v-30/3.6)
        
        return reward

    def spawn_random_veh(self, stopped_time_thres=50):
        global x_max
        global x_min
        global y_max
        global y_min
        global center_transform
        
        blueprints_ori = self.world.get_blueprint_library().filter('vehicle.*')
        spawn_points_ori = self.world.get_map().get_spawn_points()
        
        blueprints = [x for x in blueprints_ori if int(x.get_attribute('number_of_wheels')) == 4]
        blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
        blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
        blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
        blueprints = [x for x in blueprints if not x.id.endswith('t2')]

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor
        synchronous_master = True
        
        batch = []
        max_agents = randint(50,50) 
        # recommended_points = [2,3,13,14,153,154,77,78,51,52,65,66,85,86,199,200,71,72,89,90,93,94,166,168,175,181,116,117] #dns
        actor_list = self.world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        
        num_agents = len(vehicle_list)
        
        added_vehicle_num = max_agents - num_agents
        if added_vehicle_num > 1:
            added_vehicle_num = 1
        
        count = 0
        while len(batch) < added_vehicle_num:  
            #transform = spawn_points_ori[random.choice(recommended_points)]
            transform = random.choice(spawn_points_ori)
            if (transform.location.x > x_max) or (transform.location.x < x_min) or (transform.location.y > y_max) or (transform.location.y < y_min):
                continue
            too_closed_to_ego = False
            min_d = 100
            for vehicle in vehicle_list:
                d = vehicle.get_location().distance(transform.location)
                if vehicle.attributes['role_name'] == "hero" and d < 50:
                    too_closed_to_ego = True
                    break
                if d < min_d:
                    min_d = d
                if min_d < 5:
                    break
            if min_d < 5 or too_closed_to_ego == True:
                continue

            # blueprint = random.choice(blueprints)
            # if blueprint.has_attribute('color'):
            #     color = '0,0,255'
            #     blueprint.set_attribute('color', color)
            # if blueprint.has_attribute('driver_id'):
            #     driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            #     blueprint.set_attribute('driver_id', driver_id)
            # blueprint.set_attribute('role_name', 'autopilot')
            batch.append(SpawnActor(self.env_vehicle_bp, transform).then(SetAutopilot(FutureActor, True)))            
            if count > 50:
                break
        try:  
            self.client.apply_batch_sync(batch, synchronous_master)
        except:
            pass
 
        for vehicle in vehicle_list:
            if vehicle.attributes['role_name'] == "hero":
                continue
            
            self.tm.ignore_lights_percentage(vehicle, 100)
            velocity = vehicle.get_velocity()
            if vehicle.get_location().distance(circle_center) > 100:
                if vehicle.is_alive:
                    vehicle.destroy()
            if stopped_time[vehicle.id] >= 0:
                if abs(velocity.x) < 0.05 and abs(velocity.y) < 0.05:
                    stopped_time[vehicle.id] = stopped_time[vehicle.id] + 1
                else:
                    stopped_time[vehicle.id] = 0

            if stopped_time[vehicle.id] > stopped_time_thres:
                # print("Delete vehicle stay too long")
                stopped_time[vehicle.id] = -100000
                if vehicle.is_alive:
                    vehicle.destroy()

    def spawn_ego_veh(self):
        global start_point
        
        if self.ego_vehicle is not None:
            self.ego_collision_sensor.destroy()
            while self.ego_vehicle.is_alive:
                self.ego_vehicle.destroy()
            
        actor_list = self.world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        for vehicle in vehicle_list:
            d = vehicle.get_location().distance(start_point.location)
            # if vehicle.attributes['role_name'] != "hero" and d < 20:
            if vehicle.is_alive:
                vehicle.destroy()
        

        try:
            self.ego_vehicle = self.world.spawn_actor(self.ego_vehicle_bp, start_point)
        except:
            return False
        
        self.ego_collision_sensor = self.world.spawn_actor(self.ego_collision_bp, Transform(), self.ego_vehicle, carla.AttachmentType.Rigid)
        self.ego_collision_sensor.listen(lambda event: self.ego_vehicle_collision(event))
        self.ego_vehicle_collision_sign = False
        self.ego_vehicle.set_target_velocity(carla.Vector3D(0,-0,0))
        return True



        




