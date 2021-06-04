import torch as T
from ReplayBuffer import ReplayBuffer
from QNetworkCNN import QNetworkCNN
import numpy as np
import gym_tetris
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import SIMPLE_MOVEMENT

class Agent:
    def __init__(self, batch_size=32, frame_shape=(20,10), max_buffer_size=1000000, epsilon=1, gamma=0.99, lr=0.00025, epsilon_decay=9e-7
                 , min_epsilon=0.1, n_actions=6, training=True):
        
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

        self.log = T.zeros((50000000//4, 2), device = self.device)
        
            
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
        
    def train(self, epochs, idx):

      for _ in range(epochs):

        states, next_states, actions, rewards, is_done = self.replay_buffer.sample(self.batch_size)

        Q_values = self.predict_QNetwork(states.type(T.cuda.FloatTensor))

        next_Q_values = self.predict_QNetwork(next_states.type(T.cuda.FloatTensor))
        next_Q_state_values = self.target_QNetwork(next_states.type(T.cuda.FloatTensor))

        Q_value = Q_values.gather(1, actions.unsqueeze(1).type(T.int64)).squeeze(1)

        next_q_value = next_Q_state_values.gather(1, T.max(next_Q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = rewards + self.gamma * next_q_value * (1 - is_done)
        
        self.log[idx][0] = Q_value.mean()
        self.log[idx][1] = expected_q_value.mean()

        loss = self.criterion(Q_value, expected_q_value.detach()) #no gradients needed for expected values
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()          

    def playAndTrain(self, steps_limit=0):

        self.preFillReplayBuffer()
        print('preFill done !')

        env = gym_tetris.make('TetrisA-v2')
        env = JoypadSpace(env, SIMPLE_MOVEMENT)

        total_ctr = 0
        while total_ctr//24 < 50000000 :
            start = self.preProcessFrame(env.reset())
            tmp_state = [start, start, start, start]

            done = False
            step_ctr = 0 
            #self.predict_QNetwork.eval()
            total_reward = 0

            while not done:
                
                if step_ctr % 24 == 0:
                    action = self.chooseAction(tmp_state)
                    state, reward, done, _ = env.step(action)
                    state = self.preProcessFrame(state)
                    self.replay_buffer.store(state, action, reward, done)
                    self.decay_epsilon()

                    tmp_state.pop(0)
                    tmp_state.append(state)

                    if step_ctr % 96 == 0:
                        #self.predict_QNetwork.train()
                        self.train(1, min(50000000//4, total_ctr//96))
                        #self.predict_QNetwork.eval()

                else:
                    state, reward, done, _ = env.step(action)
                    if done:
                        self.replay_buffer.store(self.preProcessFrame(state), action, reward, done)

                if total_ctr % 240000 == 0:
                    state_dict = self.predict_QNetwork.state_dict()
                    self.target_QNetwork.load_state_dict(state_dict)

                    T.save(state_dict, f'weights/weights_{total_ctr}.pt')
                    T.save(self.log, 'Qlogs')
                    print('Saved weights and Qlogs')
                        
                total_reward += reward
                
                step_ctr += 1
                total_ctr  += 1
            
            print(f'Played {step_ctr} frames game with score {total_reward}, total frames {total_ctr}')

        env.close()
        
    def preProcessFrame(self, frame):
        frame = np.copy(frame[47:209, 95:176, :])
        frame = (frame[..., :3] @ [0.299, 0.587, 0.114]).astype(np.bool)
        idcs_y = np.arange(4,81,8)
        idcs_x = np.arange(4,162,8)
        frame = frame[idcs_x[:, np.newaxis], idcs_y]
        
        return T.from_numpy(frame)


    def preFillReplayBuffer(self, n_states=12500):

        env = gym_tetris.make('TetrisA-v2')
        env = JoypadSpace(env, SIMPLE_MOVEMENT)

        total_ctr = 0
      
        while total_ctr < n_states:
            state = env.reset()
            done = False
            steps = 0

            start = self.preProcessFrame(env.reset())
            tmp_state = [start, start, start, start]

            while not done:

                
                if steps % 24 == 0:
                    action = self.chooseAction(tmp_state)
                    state, reward, done, _  = env.step(action)
                    state = self.preProcessFrame(state)
                    self.replay_buffer.store(state, action, reward, done)

                    tmp_state.pop(0)
                    tmp_state.append(state)
                else:
                    state, reward, done, _ = env.step(action)
                    if done:
                        self.replay_buffer.store(self.preProcessFrame(state), action, reward, done)

                steps += 1
            total_ctr += steps//24

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

