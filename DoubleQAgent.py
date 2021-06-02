import torch as T
from ReplayBuffer import ReplayBuffer
from QNetworkCNN import QNetworkCNN
import numpy as np
import gym_tetris
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import SIMPLE_MOVEMENT

class Agent:
    def __init__(self, batch_size=32, frame_shape=(20,10), max_buffer_size=1000000, epsilon=1, gamma=0.99, lr=0.00025, epsilon_decay=9e-7
                 , min_epsilon=0.1, n_actions=5, training=True):
        
        self.frame_shape = frame_shape
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.max_buffer_size = max_buffer_size
        self.gamma = gamma

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")    
        
        self.n_actions = n_actions
        
        self.predict_QNetwork = QNetworkCNN(action_dim=n_actions).to(self.device)
        self.replay_buffer = ReplayBuffer(max_buffer_size, frame_shape, device=self.device)

        if training:
            self.lr = lr
            self.batch_size = batch_size
            self.target_QNetwork = QNetworkCNN(action_dim=n_actions).to(self.device)

            for parameter in self.target_QNetwork.parameters():
                parameter.requires_grad = False
            self.target_QNetwork.load_state_dict(self.predict_QNetwork.state_dict())
            
            self.optimizer = T.optim.Adam(self.predict_QNetwork.parameters(), lr=lr)
            #self.scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
            self.criterion = T.nn.MSELoss()
        else:
            for parameter in self.predict_QNetwork.parameters():
                parameter.requires_grad = False

            
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.min_epsilon)
    
    def chooseAction(self, state, use_epsilon=True):

      if not use_epsilon or np.random.random() > self.epsilon:
        state = T.stack(state).to(self.device) #TODO
        with T.no_grad():
            predictions = self.predict_QNetwork(state.type(T.cuda.FloatTensor).unsqueeze(0))
        return T.argmax(predictions).item() #TODO
        
      else:
        return np.random.randint(self.n_actions)
        
    def train(self, epochs):

      for _ in range(epochs):

        states, next_states, actions, rewards, is_done = self.replay_buffer.sample(self.batch_size)

        Q_values = self.predict_QNetwork(states.type(T.cuda.FloatTensor))

        next_Q_values = self.predict_QNetwork(next_states.type(T.cuda.FloatTensor))
        next_Q_state_values = self.target_QNetwork(next_states.type(T.cuda.FloatTensor))

        Q_value = Q_values.gather(1, actions.unsqueeze(1).type(T.int64)).squeeze(1)

        next_q_value = next_Q_state_values.gather(1, T.max(next_Q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = rewards + self.gamma * next_q_value * (1 - is_done)

        loss = self.criterion(Q_value, expected_q_value.detach()) #no gradients needed for expected values
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()          

    def playAndTrain(self, steps_limit=0):

        self.preFillReplayBuffer()
        print('preFill done !')

        env = gym_tetris.make('TetrisA-v0')
        env = JoypadSpace(env, SIMPLE_MOVEMENT)

        total_ctr = 0
        while total_ctr < 50000000 :
            start = self.preProcessFrame(env.reset())
            tmp_state = [start, start, start, start]

            done = False
            step_ctr = 1
            #self.predict_QNetwork.eval()
            total_reward = 0 

            while not done:
                
                action = self.chooseAction(tmp_state)
                state, reward, done, _ = env.step(action)

                #env.render()
                state = self.preProcessFrame(state)
                self.replay_buffer.store(state, action, reward, done)
                
                if step_ctr % 4 == 0:
                    #self.predict_QNetwork.train()
                    self.train(1)
                    #self.predict_QNetwork.eval()
                    
                    if step_ctr % 10000 == 0:
                        state_dict = self.predict_QNetwork.state_dict()
                        self.target_QNetwork.load_state_dict(state_dict)
                        T.save(state_dict, f'weights_{total_ctr}.pt')
                        print('Saved weights')
                        
                total_reward += reward
                self.decay_epsilon()
                step_ctr += 1

                tmp_state.pop(0)
                tmp_state.append(state)

            total_ctr  += step_ctr-1
            print(f'Played {step_ctr} frames game with score {total_reward}, total frames {total_ctr}')

        env.close()
        
    def preProcessFrame(self, frame):
        frame = np.copy(frame[47:209, 95:176, :])
        frame = (frame[..., :3] @ [0.299, 0.587, 0.114]).astype(np.uint8)
        idcs_y = np.arange(4,81,8)
        idcs_x = np.arange(4,162,8)
        frame = frame[idcs_x[:, np.newaxis], idcs_y]
        
        return T.from_numpy(frame)


    def preFillReplayBuffer(self, n_steps=50000):

        env = gym_tetris.make('TetrisA-v0')
        env = JoypadSpace(env, SIMPLE_MOVEMENT)

        total_ctr = 0 
      
        while total_ctr < n_steps:
            state = env.reset()
            done = False

            while not done:
                action = self.chooseAction(state)
                state, reward, done, _ = env.step(action)
                self.replay_buffer.store(self.preProcessFrame(state), action, reward, done)
                total_ctr += 1

        env.close()
        

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

AI = Agent()
AI.playAndTrain()

