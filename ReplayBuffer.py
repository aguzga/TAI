import torch as T
import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, input_shape): #device?
        self.size_ctr = 0
        self.max_size = max_size

        self.states = T.zeros((max_size, *input_shape), dtype=T.uint8) #TODO: uint8
        self.rewards = T.zeros((max_size, 4), dtype=T.int32)
        self.actions = T.zeros((max_size, 4), dtype=T.uint8)
        self.is_dones = T.zeros((max_size, 4), dtype=T.uint8)


    def store(self, state, action, reward, done):
        idx = self.size_ctr % self.max_size

        #if not done.any():
        self.states[idx] = state
          
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.is_dones[idx] = done
        self.size_ctr += 1

    def sample(self, batch_size):

        max_idx = min(self.max_size, self.size_ctr) - 1
        idcs = np.random.choice(max_idx, batch_size, replace=False)

        states = self.state[idcs]
        next_states = self.states[idcs+1]
        actions = self.actions[idcs]
        rewards = self.rewards[idcs]
        is_dones = self.is_done[idcs]

        return states, next_states, actions, rewards, is_dones
    
    def getData(self):
        return self.states[:self.size_ctr], self.actions[:self.size_ctr],self.rewards[:self.size_ctr], self.is_dones[:self.size_ctr]