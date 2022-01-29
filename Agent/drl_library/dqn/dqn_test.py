
import math, random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

from Agent.drl_library.dqn.replay_buffer import NaivePrioritizedBuffer, Replay_Buffer
from Agent.zzz.JunctionTrajectoryPlanner import JunctionTrajectoryPlanner
from Agent.zzz.controller import Controller
from Agent.zzz.dynamic_map import DynamicMap
from Planning_library.trustset import TrustHybridset

USE_CUDA = False#torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


class Q_network(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(Q_network, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        self.num_actions = num_actions
        
    def forward(self, x):
        return self.layers(x)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(self.num_actions)
        return action
    
    def act_hybrid(self, state, rule_action, TS):
        
        act = TS.hybrid_act(state, rule_action)
        
        return act
    
class DQN():
    def __init__(self, env, batch_size):
        self.env = env
        self.current_model = Q_network(env.observation_space.shape[0], env.action_space.n)
        self.target_model  = Q_network(env.observation_space.shape[0], env.action_space.n)

        if USE_CUDA:
            self.current_model = self.current_model.cuda()
            self.target_model  = self.target_model.cuda()
        self.device = torch.device('cuda' if USE_CUDA else 'cpu')
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.current_model.parameters())
        self.replay_buffer = NaivePrioritizedBuffer(1000000)

        self.TS = TrustHybridset(state_dimension = 10, 
                                action_num = 11,
                                save_new_data = False,
                                create_new_train_file = False)
    
    def test(self, load_step, num_frames, gamma):
        all_rewards = []
        episode_reward = 0
        
        self.load(load_step)
        self.TS.update_value_fn(self.current_model, self.target_model)
        
        # Create Agent
        trajectory_planner = JunctionTrajectoryPlanner()
        controller = Controller()
        dynamic_map = DynamicMap()

        obs, obs_ori = self.env.reset()

        while True:
            obs_ori = np.array(obs_ori)
            obs = np.array(obs)
            
            dynamic_map.update_map_from_obs(obs_ori, self.env)
            rule_trajectory, rule_action = trajectory_planner.trajectory_update(dynamic_map)

            hybrid_action = self.current_model.act_hybrid(obs, rule_action, self.TS)
            rule_trajectory = trajectory_planner.trajectory_update_CP(hybrid_action, rule_trajectory)

            control_action =  controller.get_control(dynamic_map,  rule_trajectory.trajectory, rule_trajectory.desired_speed)
            action = [control_action.acc, control_action.steering]
            new_obs, reward, done, new_obs_ori = self.env.step(action)

            obs = new_obs
            obs_ori = new_obs_ori
            episode_reward += reward

            if done:
                obs, obs_ori = self.env.reset()
                trajectory_planner.clear_buff(clean_csp=True)

                all_rewards.append(episode_reward)
                episode_reward = 0
        
    def load(self, load_step):
        try:
            self.current_model.load_state_dict(
            torch.load('saved_model/current_model_%s.pt' % (load_step))
            )

            self.target_model.load_state_dict(
            torch.load('saved_model/target_model_%s.pt' % (load_step))
            )
            
            self.replay_buffer = torch.load('saved_model/replay_buffer_%s.pt' % (load_step))
        
            print("[DQN] : Load learned model successful, step=",load_step)
        except:
            load_step = 0
            print("[DQN] : No learned model, Creat new model")
        return load_step