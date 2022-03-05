#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
import numpy as np
import random
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense 
from tensorflow.keras.activations import relu, linear 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error


# In[2]:


from collections import deque
from constants import __DQN_CONST__
from constants import __NUM_EPISODES__


# In[24]:


class DQN:
    def __init__(self, env, _para):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        
        self.size_observation_space = self.observation_space.shape[0]
        
        self.lr = _para["LR"]
        self.gamma = _para["GAMMA"]
        self.epsilon = _para["EPSILON"]
        self.decay = _para["DECAY"]
        self.batch_size = _para["BATCH_SIZE"]
        self.min_epsilon = _para["MIN_EPSILON"]
        
        self.replay_memory = deque(maxlen=500000)
        self.model = self._create_model()
        
    def _create_model(self):
        model = Sequential()
        model.add(Dense(512, input_dim=self.size_observation_space, activation=relu))
        model.add(Dense(256, activation=relu))
        model.add(Dense(self.action_space.n, activation=linear))
        
        model.compile(loss=mean_squared_error,optimizer=Adam(lr=self.lr))
        print(model.summary())
        return model
    
    
    def _sample_from_replay_memory(self):
        random_sample = random.sample(self.replay_memory, self.batch_size)
        random_sample = np.array(random_sample)
        return (random_sample[:,i] for i in range(random_sample.shape[1]))
    
    def _predict_next_state_and_fit(self):
        
    
    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = env.reset()
            reward_episode = 0
            
            num_time_steps = 10
            for _ in range(num_time_steps):
#                 env.render()
                
                # Exploration v/s Exploitation
                if (np.random.random() <= self.epsilon):
                    action = random.randrange(self.action_space.n)
                else:
                    q_table_values = self.model.predict(state)
                    print(q_table_values)
                    action = np.argmax(q_table_values[0])
                    
                next_state, reward, finished, metadata = env.step(action)
                next_state = np.reshape(next_state, [1, self.size_observation_space])
                
                self.replay_memory.append((state, action, reward, next_state, finished))
                
                if (len(self.replay_memory) >= self.batch_size):
                    (states, actions, rewards, next_states, done_list) = self._sample_from_replay_memory()
                
                
                reward_episode += reward
                state = next_state
                
                if (finished):
                    print("Episode = {}, Score = {}".format(episode, reward))
                    break


# In[25]:


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    
    env.seed(21)
    np.random.seed(21)
    
    model = DQN(env, __DQN_CONST__)
    model.train(__NUM_EPISODES__)
    


# In[ ]:




