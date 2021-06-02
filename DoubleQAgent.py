import torch as T
from ReplayBuffer import ReplayBuffer
from QNetworkCNN import QNetworkCNN
import numpy as np
import gym_tetris
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import SIMPLE_MOVEMENT

class Agent:
    def __init__(self, batch_size=128, input_shape=(4,20,10), max_buffer_size=1000000, epsilon=1, gamma=0.99, lr=1e-3, epsilon_decay=0.995
                 , min_epsilon=0.01, n_actions=5, training=True):
        
        self.input_shape = input_shape
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.max_buffer_size = max_buffer_size
        self.gamma = gamma

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")    
        
        self.n_actions = n_actions
        
        self.predict_QNetwork = QNetworkCNN(action_dim=n_actions).to(self.device)
        self.replay_buffer = ReplayBuffer(max_buffer_size, input_shape)

        for parameter in self.target_QNetwork.parameters():
            parameter.requires_grad = False

        if training:
            self.lr = lr
            self.batch_size = batch_size
            self.target_QNetwork = QNetworkCNN(action_dim=n_actions).to(self.device)

            for parameter in self.target_QNetwork.parameters():
                parameter.requires_grad = False
            self.target_QNetwork.load_state_dict(self.predict_QNetwork.state_dict())
            
            self.optimizer = T.optim.Adam(self.predict_QNetwork.parameters(), lr=lr)
            #self.scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
        else:
            for parameter in self.predict_QNetwork.parameters():
                parameter.requires_grad = False

            
    def decay_epislon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
    
    def chooseAction(self, state, use_epsilon=True):

      if not use_epsilon or np.random.random() > self.epsilon:
        state = T.Tensor(state).to(self.device) #TODO
        with T.no_grad():
            predictions = self.predict_QNetwork(state)
        return T.argmax(predictions) #TODO
        
      else:
        return np.random.randint(self.n_actions)
        
    def train(self, epochs):

      for _ in range(epochs):

        states, next_states, actions, rewards, is_done = self.replay_buffer.sample(self.batch_size)

        Q_values = self.predict_QNetwork(states)

        next_Q_values = self.predict_QNetwork(next_states)
        next_Q_state_values = self.target_QNetwork(next_states)

        Q_value = Q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        next_q_value = next_Q_state_values.gather(1, T.max(next_Q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = rewards + self.gamma * next_q_value * (1 - is_done)

        loss = T.nn.MSELoss(Q_value, expected_q_value.detach()) #no gradients needed for expected values
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()          






        state_dict = self.predict_QNetwork.state_dict()
        self.target_QNetwork.load_state_dict(state_dict)

        T.save(state_dict, 'newest_weights.pt')

      
    def play(self, epochs, steps_limit=0):
        env = gym_tetris.make('TetrisA-v0')
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        
        for _ in range(epochs):
            env.reset()
            done = False
            step_ctr = 1
            
            while not done:
                idx = step_ctr % 4
                action = self.chooseAction()
                state, reward, done, _ = env.step(action)
                #env.render()
                
                self.tmp_states[idx] =  self.preProcessFrame(state)
                self.tmp_rewards[idx] = reward
                self.tmp_actions[idx] = action
                self.tmp_is_dones[idx] = done
                
                if step_ctr and idx == 0:
                    #print('if')
                    self.agent.replay_buffer.store(self.tmp_states, self.tmp_actions, self.tmp_rewards, self.tmp_is_dones)
                    
                
                step_ctr += 1

            self.train()
            self.agent.decay_epsilon()

        env.close()
        


    def preFillReplayBuffer(self, n_steps):
        #TODO: fix
      for _ in range(n_steps):

        action = self.chooseAction(state)
        state, reward, done, _ = self.env.step(action)
        self.replay_buffer.store(state, action, reward, done)

    

#TODO: put replayBuffer in VRAM

'''
    def previewAgent(self, weights_file=None):

        if weights_file:
        #T.load
          pass

        total_reward = 0
        state = self.env.reset()
        done = False
        while not done:
            state = T.Tensor(state).to(self.device)
            with T.no_grad():
                values = self.chooseAction(state)

            state, reward, done, _ = self.env.step(action)
            total_reward += reward
            self.env.render()
          
    
        return total_reward

    #def loadWeights(self, file_path):
       # self.targetQNetwork
       '''