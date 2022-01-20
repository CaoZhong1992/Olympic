
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

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


class Q_network(nn.Module):
    def __init__(self, num_inputs, num_actions, global_graph_width=3):
        super(Q_network, self).__init__()
        
        self.q_lin = nn.Linear(num_inputs, global_graph_width)
        self.k_lin = nn.Linear(num_inputs, global_graph_width)
        self.v_lin = nn.Linear(num_inputs, global_graph_width)

        self._norm_fact = 1 / math.sqrt(num_inputs)
        
        self.layers = nn.Sequential(
            nn.Linear(global_graph_width, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        self.num_actions = num_actions
        
    def forward(self, x):
        n = int(len(x[0])/5)
        x = torch.reshape(x, [n,5]).unsqueeze(0)

        query = self.q_lin(x)
        key = self.k_lin(x)
        value = self.v_lin(x)
        scores = torch.bmm(query, key.transpose(1, 2)) * self._norm_fact
        scores = nn.functional.softmax(scores, dim=-1)
        
        atten_result =  torch.bmm(scores,value)[0][0]
        return self.layers(atten_result)
    
    def ego_attention(self, x):
        # x contains n vehicle
        # query, key, value contain n number
        
        n = int(len(x[0])/5)
        x = torch.reshape(x, [n,5]).unsqueeze(0)

        query = self.q_lin(x)
        key = self.k_lin(x)
        value = self.v_lin(x)

        ego_scores = torch.bmm(query[0][0].unsqueeze(0).unsqueeze(0), key.transpose(1, 2)) * self._norm_fact
        ego_scores = nn.functional.softmax(ego_scores, dim=-1)
        ego_atten_result = torch.mul(ego_scores.transpose(2,1), value)
        
        return ego_atten_result
        
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            
            q_value = self.forward(state).unsqueeze(0)
            # q_value_temp = q_value.unsqueeze(0)
            # print("q_value", q_value_temp)
            action  = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(self.num_actions)
        return action
    
class DQN():
    def __init__(self, env, batch_size):
        self.env = env
        self.current_model = Q_network(5, env.action_space.n)
        # self.current_model = Q_network(env.observation_space.shape[0], env.action_space.n)

        self.target_model  = Q_network(5, env.action_space.n)

        if USE_CUDA:
            self.current_model = self.current_model.cuda()
            self.target_model  = self.target_model.cuda()
        self.device = torch.device('cuda' if USE_CUDA else 'cpu')
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.current_model.parameters())
        self.replay_buffer = NaivePrioritizedBuffer(1000000)
        # self.replay_buffer = Replay_Buffer(obs_shape=env.observation_space.shape,
        #     action_shape=env.action_space.shape, # discrete, 1 dimension!
        #     capacity= 1000000,
        #     batch_size= self.batch_size,
        #     device=self.device)                

        
    def compute_td_loss(self, batch_size, beta, gamma):
        for i in range(batch_size):
            state, action, reward, next_state, done, indices, weights = self.replay_buffer.sample(1, beta) 

            state      = Variable(torch.FloatTensor(np.float32(state)))
            next_state = Variable(torch.FloatTensor(np.float32(next_state)))
            action     = Variable(torch.LongTensor(action))
            reward     = Variable(torch.FloatTensor(reward))
            done       = Variable(torch.FloatTensor(done))
            weights    = Variable(torch.FloatTensor(weights))

            q_values      = self.current_model(state)
            next_q_values = self.target_model(next_state)
            q_value          = (q_values.unsqueeze(0)).gather(1, action.unsqueeze(1)).squeeze(1)
            next_q_value     = next_q_values.unsqueeze(0).max(1)[0]
            expected_q_value = reward + gamma * next_q_value * (1 - done)
            loss  = (q_value - expected_q_value.detach()).pow(2) * weights
            prios = loss + 1e-5
            loss  = loss.mean()
                
            self.optimizer.zero_grad()
            loss.backward()
            self.replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
            self.optimizer.step()
        
        return loss
    
    def update_target(self, current_model, target_model):
        target_model.load_state_dict(current_model.state_dict())
    
    def epsilon_by_frame(self, frame_idx):
        epsilon_start = 1.0
        epsilon_final = 0.01
        epsilon_decay = 500
        return epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
    
    def beta_by_frame(self, frame_idx):
        beta_start = 0.4
        beta_frames = 1000  
        return min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)
    
    def train(self, load_step, num_frames, gamma):
        losses = []
        all_rewards = []
        episode_reward = 0
        
        self.load(load_step)
        
        # Create Agent
        trajectory_planner = JunctionTrajectoryPlanner()
        controller = Controller()
        dynamic_map = DynamicMap()

        obs = self.env.reset()
        for frame_idx in range(load_step, load_step+num_frames + 1):

            obs = np.array(obs)
            dynamic_map.update_map_from_obs(obs, self.env)
            
            # Dqn
            epsilon = self.epsilon_by_frame(frame_idx)
            dqn_action = self.current_model.act(obs, epsilon)
            obs_tensor = Variable(torch.FloatTensor(np.float32(obs))).unsqueeze(0)
            ego_attention = self.current_model.ego_attention(obs_tensor).detach().numpy()
            
            rule_trajectory, action = trajectory_planner.trajectory_update(dynamic_map)
            rule_trajectory = trajectory_planner.trajectory_update_CP(dqn_action, rule_trajectory)
            # trajectory_planner.generation_control_signal_of_action(1, dynamic_map)
            # # Control
            
            control_action =  controller.get_control(dynamic_map,  rule_trajectory.trajectory, rule_trajectory.desired_speed)
            action = [control_action.acc, control_action.steering]

            new_obs, reward, done, _ = self.env.step(action, ego_attention = ego_attention)
            # print("[DQN]: ----> RL Action",dqn_action)

            # self.replay_buffer.add(obs, np.array([dqn_action]), np.array([reward]), new_obs, np.array([done]))
            self.replay_buffer.push(obs, dqn_action, reward, new_obs, done)
            
            obs = new_obs
            episode_reward += reward
            
            if done:
                obs = self.env.reset()
                trajectory_planner.clear_buff(clean_csp=True)

                all_rewards.append(episode_reward)
                episode_reward = 0
                
            if (frame_idx) > self.batch_size:
                beta = self.beta_by_frame(frame_idx)
                loss = self.compute_td_loss(self.batch_size, beta, gamma)
                
                # losses.append(loss.data[0])
                
            # if frame_idx % 200 == 0:
            #     plot(frame_idx, all_rewards, losses)
                
            if (frame_idx) % 10000 == 0:
                self.update_target(self.current_model, self.target_model)
                self.save(frame_idx)

    def save(self, step):
        torch.save(
            self.current_model.state_dict(),
            'saved_model/current_model_%s.pt' % (step)
        )
        torch.save(
            self.target_model.state_dict(),
            'saved_model/target_model_%s.pt' % (step)
        )
        torch.save(
            self.replay_buffer,
            'saved_model/replay_buffer_%s.pt' % (step)
        )
        
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