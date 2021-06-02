from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import SIMPLE_MOVEMENT
from DoubleQAgent import Agent
import numpy as np
import torch as T
import pickle

class GameCollector:
    
    def __init__(self, weights_file):
        self.agent = Agent(training=False, max_buffer_size=10000)
        self.weights_file = weights_file

        self.tmp_states = T.zeros(self.agent.input_shape, dtype=T.uint8) 
        self.tmp_rewards = T.zeros(4, dtype=T.int32)
        self.tmp_actions = T.zeros(4, dtype=T.uint8)
        self.tmp_is_dones = T.zeros(4, dtype=T.uint8)
        
    def preProcessFrame(self, frame):
        frame = np.copy(frame[47:209, 95:176, :])
        frame = (frame[..., :3] @ [0.299, 0.587, 0.114]).astype(np.uint8)
        idcs_y = np.arange(4,81,8)
        idcs_x = np.arange(4,162,8)
        frame = frame[idcs_x[:, np.newaxis], idcs_y]
        
        return T.from_numpy(frame)
    
    def play(self, epochs, steps_limit=0):
        env = gym_tetris.make('TetrisA-v0')
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        
        for _ in range(epochs):
            env.reset()
            done = False
            step_ctr = 1
            
            while not done:
                idx = step_ctr % 4
                action = env.action_space.sample() #self.agent.chooseAction()
                state, reward, done, _ = env.step(action)
                env.render()
                
                self.tmp_states[idx] =  self.preProcessFrame(state)
                self.tmp_rewards[idx] = reward
                self.tmp_actions[idx] = action
                self.tmp_is_dones[idx] = done
                
                
                if step_ctr and idx == 0:
                    #print('if')
                    self.agent.replay_buffer.store(self.tmp_states, self.tmp_actions, self.tmp_rewards, self.tmp_is_dones)
                
                step_ctr += 1
                
            self.agent.decay_epsilon()

        env.close()
        
    
    def run(self):   
        self.target_QNetwork.load_state_dict(self.weights_file)
        self.play(1)
        #pickle.dump(self.agent.replay_buffer.getData(), open( "test", "wb" ))
        #np.save('test', self.agent.replay_buffer.getData())

        #a = pickle.load(open( "test", "rb" ))
        #for obj in a:
            #print(obj.size())
        #print(a[2].sum())
a = GameCollector('no')
a.run()
