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
try:
	sys.path.append(glob.glob("/home/icv/.local/lib/python3.6/site-packages/")[0])
except IndexError:
	pass

import carla
import time
import numpy as np
import math
import random
import gym
import matplotlib.pyplot as plt

from tqdm import tqdm
from Test_Scenarios.TestScenario_CarEnv_05_Round import CarEnv_05_Round
from Agent.zzz.JunctionTrajectoryPlanner import JunctionTrajectoryPlanner
from Agent.zzz.controller import Controller
from Agent.zzz.dynamic_map import DynamicMap
from Agent.drl_library.dqn.dqn import DQN

# from Agent.zzz.CP import CP, Imagine_Model
EPISODES=2642

if __name__ == '__main__':

    # Create environment
    
    env = CarEnv_05_Round()

    model = DQN(env, batch_size=2)
    model.train(num_frames=10000,  gamma=0.99)
    # model.save("dqn_cartpole")
        

