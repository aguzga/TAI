import torch as T
import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, input_shape, device): #device?
        self.size_ctr = 0
        self.max_size = max_size

        self.states = T.zeros((max_size, *input_shape), dtype=T.uint8, device=device)
        self.rewards = T.zeros(max_size, dtype=T.int32, device=device)
        self.actions = T.zeros(max_size, dtype=T.uint8, device=device)
        self.is_dones = T.zeros(max_size, dtype=T.uint8, device=device)


    def store(self, state, action, reward, done):
        idx = self.size_ctr % self.max_size

        #if not done.any():
        self.states[idx] = state
          
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.is_dones[idx] = done
        self.size_ctr += 1

    def sample(self, batch_size):

        max_idx = min(self.max_size, self.size_ctr) - 5
        idcs = np.random.choice(max_idx, batch_size, replace=False)

        states, next_states = [], []

        for i in range(4):
            states.append(self.states[idcs+i])
            next_states.append(self.states[idcs+i+1])
        
        states = T.stack(states, dim=1)
        next_states = T.stack(next_states, dim=1)

        actions = self.actions[idcs]
        rewards = self.rewards[idcs]
        is_dones = self.is_dones[idcs]

        return states, next_states, actions, rewards, is_dones
    
    def getData(self):
        return self.states[:self.size_ctr], self.actions[:self.size_ctr],self.rewards[:self.size_ctr], self.is_dones[:self.size_ctr]