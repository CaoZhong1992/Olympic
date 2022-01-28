import os
import os.path as osp
import random
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

        self.current_model = None
        self.traget_model = None
        self.rule_policy = None

    
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

    def update_value_fn(self, current_model, target_model):

        self.current_model = current_model
        self.traget_model = target_model

    def TS_ConfidenceValue(self, state, action):

        state_torch      = Variable(torch.FloatTensor(np.float32([state])))
        action_torch     = Variable(torch.LongTensor([action]))
        
        q_value_s = self.current_model(state_torch).unsqueeze(0)        
        q_value_sa = q_value_s.gather(1, action_torch.unsqueeze(1)).squeeze(1)

        if self.in_TS(state, action):
            return q_value_sa[action]
        
        return -10000

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
          
        if action == self.rule_policy(state):
            return True
        
        return False
    
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
