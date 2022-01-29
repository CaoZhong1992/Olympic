from itertools import accumulate
import os
import os.path as osp
import math
import numpy as np
from rtree import index as rindex
from collections import deque

import torch

class Trustset(object):
    
    def __init__(self,
                 visited_times_thres = 30,
                 is_training = True,
                 debug = True,
                 save_new_data = True,
                 create_new_train_file = True,
                 create_new_record_file = True,
                 save_new_driving_data = True):

        self.visited_times_thres = visited_times_thres
        self.is_training = is_training
        self.trajectory_buffer = deque(maxlen=20)
        self.debug = debug
        self.save_new_data = save_new_data
        self.create_new_train_file = create_new_train_file
        self.create_new_record_file = create_new_record_file
        self.save_new_driving_data = save_new_driving_data
        self.state_dimension = 3
        self.action_num = 3
        self.gamma = 0.95
        self._setup_data_saving()
    
    def _setup_data_saving(self):
        
        if self.create_new_train_file:
            if osp.exists("trustset/state_index.dat"):
                os.remove("trustset/state_index.dat")
                os.remove("trustset/state_index.idx")
            if osp.exists("trustset/visited_state.txt"):
                os.remove("trustset/visited_state.txt")
            if osp.exists("trustset/visited_value.txt"):
                os.remove("trustset/visited_value.txt")
            # self.visited_state_value = []
            self.visited_state_counter = 0
        else:
            # self.visited_state_value = np.loadtxt("trustset/visited_value.txt")
            # self.visited_state_value = self.visited_state_value.tolist()
            self.visited_state_counter = len(self.visited_state_value)

        self.visited_state_outfile = open("trustset/visited_state.txt", "a")
        self.visited_state_format = " ".join(("%f",)*(self.state_dimension+1))+"\n"

        # self.visited_value_outfile = open("trustset/visited_value.txt", "a")
        # self.visited_value_format = " ".join(("%f",)*2)+"\n"

        visited_state_tree_prop = rindex.Property()
        visited_state_tree_prop.dimension = self.state_dimension+1
        
        # ego_state, action
        self.visited_state_dist = np.array([[1, 1, 1, 0.5]])
        self.visited_state_tree = rindex.Index('trustset/state_index',properties=visited_state_tree_prop)

        # if self.create_new_record_file:
        #     if osp.exists("driving_record.txt"):
        #         os.remove("driving_record.txt")
        # self.driving_record_outfile = open("driving_record.txt","a")
        # self.driving_record_format = " ".join(("%f",)*(self.state_dimension+9))+"\n"

    def in_TS(self, state, action = None):
        
        if action is None:
            return self._in_TS_state(state)
        
        if self._is_trust_action_by_rules(state,action):
            return True
 
        if self._is_trust_action_by_data(state,action):
            return True
        
        return False
    
    def _in_TS_state(self, state):
        
        for act in range(self.action_num):
            if self._is_trust_action_by_data(state,act):
                return True
            
        return False
    
    def get_state_num(self, state):
        state_action_num = []
        for act in range(self.action_num):
            state_action_num.append(self._get_state_action_num(state,act))
        
        return np.array(state_action_num)
    
    def _get_state_action_num(self, state, action):
        
        return self._calculate_visited_times(state, action)
    
    def _is_trust_action_by_rules(self, state, action):
          
        if int(action) == 2:
            return True
    
    def _is_trust_action_by_data(self, state, action):
        
        if self._calculate_visited_times(state,action)>0:
            return True
    
    def _calculate_visited_times(self, state, action):
        
        state_with_action = np.append(state,action)
        return sum(1 for _ in self.visited_state_tree.intersection(state_with_action.tolist()))
    
    def add_data(self, state, action, reward):
        
        if self.save_new_data:
            state_with_action = np.append(state,action)
            # self.visited_state_value.append([action,reward])
            self.visited_state_tree.insert(self.visited_state_counter,
                tuple((state_with_action-self.visited_state_dist).tolist()[0]+(state_with_action+self.visited_state_dist).tolist()[0]))
            self.visited_state_outfile.write(self.visited_state_format % tuple(state_with_action))
            # self.visited_value_outfile.write(self.visited_value_format % tuple([state_with_action, reward]))
            self.visited_state_counter += 1


class TrustHybridset(object):

    def __init__(self,
                 state_dimension,
                 action_num,
                 visited_times_thres = 30,
                 save_new_data = True,
                 create_new_train_file = True,
                 create_new_record_file = True,
                 save_new_driving_data = True):

        self.visited_times_thres = visited_times_thres
        self.save_new_data = save_new_data
        self.create_new_train_file = create_new_train_file
        self.create_new_record_file = create_new_record_file
        self.save_new_driving_data = save_new_driving_data
        self.state_dimension = state_dimension
        self.action_num = action_num
        self.gamma = 0.95
        self._setup_data_saving()

        self.current_model = None
        self.traget_model = None
        self.rule_policy = None

        self.accumulated_rewards = []

    def _setup_data_saving(self):
        
        if self.create_new_train_file:
            if osp.exists("trustset/state_index.dat"):
                os.remove("trustset/state_index.dat")
                os.remove("trustset/state_index.idx")
            if osp.exists("trustset/visited_state.txt"):
                os.remove("trustset/visited_state.txt")
            if osp.exists("trustset/visited_value.txt"):
                os.remove("trustset/visited_value.txt")
            self.visited_state_value = []
            self.visited_state_counter = 0
        else:
            self.visited_state_value = np.loadtxt("trustset/visited_value.txt")
            self.visited_state_value = self.visited_state_value.tolist()
            self.visited_state_counter = len(self.visited_state_value)

        self.visited_state_outfile = open("trustset/visited_state.txt", "a")
        self.visited_state_format = " ".join(("%f",)*(self.state_dimension+1))+"\n"

        self.visited_value_outfile = open("trustset/visited_value.txt", "a")
        self.visited_value_format = " ".join(("%f",)*1)+"\n"

        visited_state_tree_prop = rindex.Property()
        visited_state_tree_prop.dimension = self.state_dimension+1
        
        # ego_state, action
        self.visited_state_dist = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5]])
        self.visited_state_tree = rindex.Index('trustset/state_index',properties=visited_state_tree_prop)

    def update_value_fn(self, current_model, target_model):

        self.current_model = current_model
        self.traget_model = target_model

    def hybrid_act(self, state):
        rule_action = self.rule_policy(state)

        action_value = []

        for act in range(self.action_num):
            if act == rule_action:
                action_value.append(self._get_state_value(state, rule_action))
            else:
                action_value.append(self.HoeffidingLCB(state, act))

        return np.argmax(np.array(action_value))

    # def TS_ConfidenceValue(self, state, action):

    #     # Hoeffiding
    #     return self.HoeffidingLCB(state, action)

    #     # Central Limitation
    #     return self.GaussianLCB(state, action)
    

    def _get_state_value(self, state, action):
        state_torch      = Variable(torch.FloatTensor(np.float32([state])))
        action_torch     = Variable(torch.LongTensor([action]))
        
        q_value_s = self.current_model(state_torch).unsqueeze(0)        
        q_value_sa = q_value_s.gather(1, action_torch.unsqueeze(1)).squeeze(1)

        return q_value_sa.detach().numpy()[0]

    def _is_trust_action_by_rules(self, state, action):
          
        if action == self.rule_policy(state):
            return True
        
        return False

    def HoeffidingLCB(self, state, action):

        n = self._get_state_action_num(state, action)
        state_value = self._get_state_value(state, action)
        H_error = self._HoeffidingError(n)

        return state_value - H_error

    def _HoeffidingError(self, n, a = 0, b = 20, alpha = 0.1):
        """
        b = 20 # Max Value
        a = 0 # Min Value
        alpha = 0.1 # 1-alpha confidence
        """
        if n == 0:
            return b - a
        else:
            return (b-a)*math.sqrt(math.log(1/alpha)/(2*n))

    def GaussianNaiveLCB(self, state, action, thres_num = 30, min_value = -20):

        if self._calculate_visited_times(state,action) > thres_num:
            return self._get_state_value(state, action)

        return min_value

    def GaussianLCB(self, state, action, thres_num = 30, min_value = -20):

        if self._calculate_visited_times(state,action) < thres_num:
            return min_value

        value_array = np.array(self._get_state_action_values)
        
        mean = np.mean(value_array)
        var = np.var(value_array)
        sigma = np.sqrt(var)

        return min_value
    
    def _calculate_visited_times(self, state, action):
        
        state_with_action = np.append(state,action)
        return sum(1 for _ in self.visited_state_tree.intersection(state_with_action.tolist()))

    def _get_state_action_values(self, state, action):

        state_with_action = np.append(state,action)
        return [self.visited_state_value[idx] for idx in self.visited_state_tree.intersection(state_with_action.tolist()) if idx < len(self.visited_state_value)]

    def add_data(self, state, action):

        if self.save_new_data:
            state_with_action = np.append(state,action)
            self.visited_state_tree.insert(self.visited_state_counter,
                tuple((state_with_action-self.visited_state_dist).tolist()[0]+(state_with_action+self.visited_state_dist).tolist()[0]))
            self.visited_state_outfile.write(self.visited_state_format % tuple(state_with_action))
            self.visited_state_counter += 1

    def add_data_during_data_collection(self, state, action, reward, done):

        if self.save_new_data:
            state_with_action = np.append(state,action)
            self.accumulated_rewards.append(reward)
            self.visited_state_tree.insert(self.visited_state_counter,
                tuple((state_with_action-self.visited_state_dist).tolist()[0]+(state_with_action+self.visited_state_dist).tolist()[0]))
            self.visited_state_outfile.write(self.visited_state_format % tuple(state_with_action))
            self.visited_state_counter += 1

            horizon = 20
            if done:
                horizon = 0

            while len(self.accumulated_rewards) > horizon:
                value = sum(self.accumulated_rewards)
                self.accumulated_rewards.pop(0)

                self.visited_state_value.append([value])
                self.visited_value_outfile.write(self.visited_value_format % tuple([value]))
