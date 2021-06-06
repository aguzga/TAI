import torch as T
import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, input_shape, device): #device?
        self.size_ctr = 0
        self.max_size = max_size

        self.next_states = T.zeros((max_size, *input_shape), device=device)
        self.states = T.zeros((max_size, *input_shape), device=device)
        self.rewards = T.zeros(max_size, device=device)
        self.is_dones = T.zeros(max_size, device=device)


    def store(self, state, reward, next_state, done):
        idx = self.size_ctr % self.max_size

        #if not done.any():
        self.states[idx] = state
        self.next_states[idx] = next_state
        self.rewards[idx] = reward
        self.is_dones[idx] = done
        self.size_ctr += 1

    def sample(self, batch_size):

        max_idx = min(self.max_size, self.size_ctr)
        idcs = np.random.choice(max_idx, batch_size, replace=False)
        next_states = self.next_states[idcs]
        states = self.states[idcs]

        rewards = self.rewards[idcs]
        is_dones = self.is_dones[idcs]

        return states, rewards, next_states, is_dones
    
   