import torch as T
from ReplayBuffer import ReplayBuffer
from QNetworkCNN import QNetworkCNN
import numpy as np
from src.tetris import Tetris
import time

import torch.nn.functional as F

class Agent:
    def __init__(self, batch_size=512, frame_shape=(4,), max_buffer_size=300000, epsilon=1, gamma=0.99, lr=1e-3, epsilon_decay=0.00006188
                 , min_epsilon=1e-2, n_actions=1, training=True):
        
        self.frame_shape = frame_shape
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.max_buffer_size = max_buffer_size
        self.gamma = gamma

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")    
        
        self.n_actions = n_actions
        
        self.predict_QNetwork = QNetworkCNN().to(self.device)

        if training:
            self.replay_buffer = ReplayBuffer(max_buffer_size, frame_shape, device=self.device)
            self.lr = lr
            self.batch_size = batch_size
            self.target_QNetwork = QNetworkCNN().to(self.device)
            
            

            for parameter in self.target_QNetwork.parameters():
                parameter.requires_grad = False
            self.target_QNetwork.eval()
            self.target_QNetwork.load_state_dict(self.predict_QNetwork.state_dict())
            
            self.optimizer = T.optim.Adam(self.predict_QNetwork.parameters(), lr=lr)
            #self.scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
            self.criterion = T.nn.MSELoss()
        else:
            for parameter in self.predict_QNetwork.parameters():
                parameter.requires_grad = False

        #self.log = T.zeros((50000000//4, 2), device = self.device)
            
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.min_epsilon)
    
    def chooseAction(self, states, use_epsilon=True):

      if not use_epsilon or np.random.random() > self.epsilon:
        states = states.to(self.device)
        self.predict_QNetwork.eval()
        with T.no_grad():
            predictions = self.predict_QNetwork(states.type(T.cuda.FloatTensor))[:, 0]
        self.predict_QNetwork.train()
        return T.argmax(predictions).item()
        
      else:
        return np.random.randint(len(states))
        
    def train(self, epochs):

      for _ in range(epochs):

        states, rewards, next_states, is_done = self.replay_buffer.sample(self.batch_size)
        rewards = rewards.reshape(512, 1)
        is_done = is_done.reshape(512, 1)
        #print(states.size())
        #print(rewards.size())
        #print(next_states.size())
        #print(is_done.size())
        


        Q_values = self.predict_QNetwork(states.type(T.cuda.FloatTensor))

        self.predict_QNetwork.eval()
        with T.no_grad():
            next_Q_values = self.predict_QNetwork(next_states.type(T.cuda.FloatTensor))
        self.predict_QNetwork.train()

        expected_q_values = rewards + self.gamma * next_Q_values * (1 - is_done)
        #print(Q_values.size(), next_Q_values.size())
        
        #self.log[idx][0] = Q_value.mean()
        #self.log[idx][1] = expected_q_value.mean()
        #loss = F.smooth_l1_loss(Q_value, expected_q_value)
        #loss = (Q_values - expected_q_values.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss = self.criterion(Q_values, expected_q_values) #no gradients needed for expected values
        #print(Q_values.size(), expected_q_values.size())
        #time.sleep(1000)
        loss.backward()
        
        #for param in self.predict_QNetwork.parameters():
            #param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

    def playAndTrain(self, steps_limit=0):
        env = Tetris(width=10, height=20, block_size=30)

        game_ctr = 0
        line_buffer = np.zeros(100)
        while game_ctr < 50000 :
            
            done = False
            state = env.reset()
            while not done:
            
                next_steps = env.get_next_states()
                next_actions, next_states = zip(*next_steps.items())
                next_states = T.stack(next_states)

                index = self.chooseAction(next_states)
                next_state = next_states[index, :]
                action = next_actions[index]

                reward, done = env.step(action, render=False)

                self.replay_buffer.store(state, reward, next_state, done)

                state = next_state

            self.decay_epsilon()

            if self.replay_buffer.size_ctr < self.max_buffer_size / 10:
                if self.replay_buffer.size_ctr % 100 == 0:
                    print(self.replay_buffer.size_ctr)
                continue
            
            self.train(1)

            #if game_ctr % 2 == 0:
                #self.target_QNetwork.load_state_dict(self.predict_QNetwork.state_dict())

            if game_ctr % 1000 == 0:
                T.save(self.predict_QNetwork, f'weights2/weights_{game_ctr}.pt')
                print('Saved weights')
            
            final_score = env.score
            #final_cleared_lines = env.cleared_lines
            
            line_buffer[game_ctr % 100] = env.cleared_lines
            if game_ctr % 100 == 0:
                print(f'Played 100 games with score {final_score}, total games {game_ctr}, avg cleared lines {line_buffer.mean()}')

            game_ctr += 1

    
if __name__ == '__main__':
    AI = Agent()
    AI.playAndTrain()

