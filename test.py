import sys
import glob

sys.path.append(glob.glob('/home/zhcao/Downloads/CARLA_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.8-linux-x86_64.egg')[0])
sys.path.append(glob.glob('/home/zhcao/Downloads/CARLA_0.9.13/PythonAPI/carla')[0])

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner


if __name__ == '__main__':

    print("hello world")