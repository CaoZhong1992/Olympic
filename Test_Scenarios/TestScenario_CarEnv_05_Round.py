import glob
import os
import sys
try:
	sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
		sys.version_info.major,
		sys.version_info.minor,
		'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
	pass

import carla
import time
import numpy as np
import math
import random
import gym
import threading
from random import randint
from carla import Location, Rotation, Transform, Vector3D, VehicleControl
from collections import deque
from tqdm import tqdm
from gym import core, error, spaces, utils
from gym.utils import seeding
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

from Agent.zzz.dynamic_map import Lanepoint, Lane, Vehicle
from Agent.zzz.tools import *
from geometry import dist_from_point_to_polyline, dense_polyline



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
        self.tm.set_random_device_seed(0)

        actors = self.world.get_actors().filter('vehicle*')
        for actor in actors:
            actor.destroy()

        # Generate Reference Path
        dao = GlobalRoutePlannerDAO(self.world.get_map(), 1)
        self.global_routing()
        global start_point
        global goal_point
        self.start_point = start_point
        self.goal_point = goal_point

        # RL settingss
        self.action_space = spaces.Discrete(9)
        self.low  = np.array([0,  0, 0, 0, 0,0,  0, 0, 0, 0], dtype=np.float64)
        self.high = np.array([1,  1, 1, 1, 1,1,  1, 1, 1, 1], dtype=np.float64)    
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float64)
        self.state_dimension = 20

        # Ego Vehicle Setting
        self.ego_vehicle_bp = random.choice(self.world.get_blueprint_library().filter('vehicle.lincoln.mkz_2020'))
        if self.ego_vehicle_bp.has_attribute('color'):
            color = '255,255,0'
            self.ego_vehicle_bp.set_attribute('color', color)
            self.ego_vehicle_bp.set_attribute('role_name', "ego_vehicle")
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

        # Control Env Vehicle
        self.has_set = np.zeros(1000000)
        self.stopped_time = np.zeros(1000000)   

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
        
        dao = GlobalRoutePlannerDAO(self.world.get_map(), 1)

        # first route
        grp = GlobalRoutePlanner(self.world.get_map(), sampling_resolution=1) # Carla 0913
        current_route = grp.trace_route(carla.Location(start.location.x,
                                                start.location.y + 20,
                                                start.location.z),
                                carla.Location(goal.location.x,
                                                goal.location.y,
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
                                                start.location.y,
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

    def ego_vehicle_stuck(self, stay_thres = 15):        
        ego_vehicle_velocity = math.sqrt(self.ego_vehicle.get_velocity().x ** 2 + self.ego_vehicle.get_velocity().y ** 2 + self.ego_vehicle.get_velocity().z ** 2)
        if ego_vehicle_velocity < 0.05:
            pass
        else:
            self.stuck_time = time.time()

        if time.time() - self.stuck_time > stay_thres:
            return True
        return False

    def ego_vehicle_pass(self):
        global goal_point
        ego_location = self.ego_vehicle.get_location()
        if ego_location.distance(goal_point.location) < 35:
            return True
        else:
            return False

    def ego_vehicle_collision(self, event):
        self.ego_vehicle_collision_sign = True

    def wrap_state(self):
        # state = [0 for i in range((OBSTACLES_CONSIDERED + 1) * 4)]
        state  = np.array([0,0,0,0,0,0,0,0,0,0], dtype=np.float64)

        ego_vehicle_state = Vehicle()
        ego_vehicle_state.x = self.ego_vehicle.get_location().x
        ego_vehicle_state.y = self.ego_vehicle.get_location().y
        ego_vehicle_state.v = math.sqrt(self.ego_vehicle.get_velocity().x ** 2 + self.ego_vehicle.get_velocity().y ** 2 + self.ego_vehicle.get_velocity().z ** 2)

        ego_vehicle_state.yaw = self.ego_vehicle.get_transform().rotation.yaw / 180.0 * math.pi # Transfer to rad
        ego_vehicle_state.yawdt = self.ego_vehicle.get_angular_velocity()

        ego_vehicle_state.vx = ego_vehicle_state.v * math.cos(ego_vehicle_state.yaw)
        ego_vehicle_state.vy = ego_vehicle_state.v * math.sin(ego_vehicle_state.yaw)

        # Ego state
        ego_ffstate = get_frenet_state(ego_vehicle_state, self.ref_path_array, self.ref_path_tangets)
        state[0] = ego_vehicle_state.x  
        state[1] = ego_vehicle_state.y 
        state[2] = ego_vehicle_state.vx 
        state[3] = ego_vehicle_state.vy 
        state[4] = ego_vehicle_state.yaw 


        actor_list = self.world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        front_vehicle = self.get_front_vehicle(self.ref_path_array, vehicle_list)
        if front_vehicle is not None:
            state[5] = front_vehicle.get_location().x
            state[6] = front_vehicle.get_location().y
            front_velocity = math.sqrt(front_vehicle.get_velocity().x ** 2 + front_vehicle.get_velocity().y ** 2 + front_vehicle.get_velocity().z ** 2)
            front_yaw = front_vehicle.get_transform().rotation.yaw / 180.0 * math.pi
            state[7] = front_velocity * math.cos(front_yaw)
            state[8] = front_velocity * math.sin(front_yaw)
            state[9] = front_yaw
            # state[2] = self.ego_vehicle.get_location().distance(front_vehicle.get_transform())
        print("state",state)

        return state

    def get_front_vehicle(self, lane_path, vehicle_list):
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
        v_id = 0
        for vehicle in vehicle_list:
            if vehicle.attributes['role_name'] == "ego_vehicle":
                continue
            vehicle_x = vehicle.get_location().x
            vehicle_y = vehicle.get_location().y
            d2lane, d2lane_head, d2lane_tail = dist_from_point_to_polyline(vehicle_x, vehicle_y, lane_path)
            if d2lane_head > ego2lane_head:
                v2lane_front.append([v_id, d2lane, d2lane_head, d2lane_tail, vehicle])
            v_id += 1
        v2lane_front = np.array(v2lane_front)
        if len(v2lane_front) > 0:
            front_vehicle_id = v2lane_front[np.argmin(v2lane_front[:, 2])][0]
            front_vehicle = v2lane_front[np.argmin(v2lane_front[:, 2])][4]
            # print("front vehicle id is:", front_vehicle_id)
            return front_vehicle
        else:
            return None

    def found_closest_obstacles_t_intersection(self, ego_ffstate):
        obs_tuples = []
        for obs in self.world.get_actors().filter('vehicle*'):
            # Calculate distance
            p1 = np.array([self.ego_vehicle.get_location().x ,  self.ego_vehicle.get_location().y])
            p2 = np.array([obs.get_location().x , obs.get_location().y])
            p3 = p2 - p1
            p4 = math.hypot(p3[0],p3[1])
            
            # Obstacles too far
            one_obs = (obs.get_location().x, obs.get_location().y, obs.get_velocity().x, obs.get_velocity().y, obs.get_transform().rotation.yaw/ 180.0 * math.pi, p4)
            if 0 < p4 < 50:
                obs_tuples.append(one_obs)
        
        closest_obs = []
        fake_obs = [0 for i in range(11)]  #len(one_obs)
        for i in range(0, OBSTACLES_CONSIDERED ,1): # 3 obs
            closest_obs.append(fake_obs)
        
        # Sort by distance
        sorted_obs = sorted(obs_tuples, key=lambda obs: obs[5])   
        for obs in sorted_obs:
            closest_obs[0] = obs 

        return closest_obs
                                            
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
        self.spawn_ego_veh()
        self.world.tick() 

        # State
        state = self.wrap_state()

        # Record
        self.record_information_txt()
        self.task_num += 1
        self.case_id += 1

        return state

    def step(self, action):
        # Control ego vehicle
        throttle = max(0,float(action[0]))  # range [0,1]
        brake = max(0,-float(action[0])) # range [0,1]
        steer = action[1] # range [-1,1]
        self.ego_vehicle.apply_control(carla.VehicleControl(throttle = throttle, brake = brake, steer = steer))
        self.spawn_random_veh()

        self.world.tick()

        # State
        state = self.wrap_state()

        reward = 0
        
        # If finish
        done = False
        if self.ego_vehicle_collision_sign:
            self.collision_num += + 1
            done = True
            reward = - 10
            print("[CARLA]: Collision!")
        
        # if self.ego_vehicle_pass():
        #     done = True
        #     reward = 1
        #     print("[CARLA]: Successful!")

        elif self.ego_vehicle_stuck():
            self.stuck_num += 1
            done = True
            reward = 0.0
            print("[CARLA]: Stuck!")

        return state, reward, done, None

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
                if vehicle.attributes['role_name'] == "ego_vehicle" and d < 50:
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
            
            # print("Spawn a vehicle, num=",len(vehicle_list))
            self.client.apply_batch_sync(batch, synchronous_master)
            if count > 50:
                break
            
        for vehicle in vehicle_list:
            self.tm.ignore_lights_percentage(vehicle, 100)
            velocity = vehicle.get_velocity()
            if vehicle.get_location().distance(circle_center) > 100:
                vehicle.destroy()
            if stopped_time[vehicle.id] >= 0:
                if abs(velocity.x) < 0.05 and abs(velocity.y) < 0.05:
                    stopped_time[vehicle.id] = stopped_time[vehicle.id] + 1
                else:
                    stopped_time[vehicle.id] = 0

            if stopped_time[vehicle.id] > stopped_time_thres:
                # print("Delete vehicle stay too long")
                stopped_time[vehicle.id] = -100000
                vehicle.destroy()

    def spawn_ego_veh(self):
        global start_point
        if self.ego_vehicle is not None:
            self.ego_collision_sensor.destroy()
            self.ego_vehicle.destroy()
            
        actor_list = self.world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        for vehicle in vehicle_list:
            d = vehicle.get_location().distance(start_point.location)
            if vehicle.attributes['role_name'] != "ego_vehicle" and d < 20:
                vehicle.destroy()

        self.ego_vehicle = self.world.spawn_actor(self.ego_vehicle_bp, start_point)
        self.ego_collision_sensor = self.world.spawn_actor(self.ego_collision_bp, Transform(), self.ego_vehicle, carla.AttachmentType.Rigid)
        self.ego_collision_sensor.listen(lambda event: self.ego_vehicle_collision(event))
        self.ego_vehicle_collision_sign = False
        self.ego_vehicle.set_target_velocity(carla.Vector3D(0,-0,0))



        




