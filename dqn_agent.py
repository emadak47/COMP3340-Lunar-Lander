#!/usr/bin/env python
# coding: utf-8

# In[4]:


import gym
import numpy as np
import random
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense 
from tensorflow.keras.activations import relu, linear 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error


# In[5]:


from collections import deque
from constants import __DQN_CONST__
from constants import __NUM_EPISODES__


# In[6]:


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
        
        states = np.array([i[0] for i in random_sample])
        actions = np.array([i[1] for i in random_sample])
        rewards = np.array([i[2] for i in random_sample])
        next_states = np.array([i[3] for i in random_sample])
        done_list = np.array([i[4] for i in random_sample])
        
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        
        return states, actions, rewards, next_states, done_list
            
    def _predict_next_state_and_fit(self, states, actions, rewards, next_states, done_list):
        q_table_values = self.model.predict_on_batch(next_states)
        q_table_targets = self.model.predict_on_batch(states)
        max_q_values_next_state = np.amax(q_table_values, axis=1)
        
        q_table_targets[np.arange(self.batch_size), actions] = rewards + self.gamma * (max_q_values_next_state) * (1 - done_list)
        self.model.fit(states, q_table_targets, verbose=0)
    
    def train(self, num_episodes):
        for episode in range(400):
            state = env.reset()
            reward_episode = 0
            
            num_time_steps = 1000
            for _ in range(num_time_steps):
#                 env.render()
                state = np.reshape(state, (1, 8))
                # Exploration v/s Exploitation
                if (np.random.random() <= self.epsilon):
                    action = random.randrange(self.action_space.n)
                else:
                    q_table_values = self.model.predict(state)
                    action = np.argmax(q_table_values[0])
                    
                next_state, reward, done, metadata = env.step(action)
                next_state = np.reshape(next_state, [1, self.size_observation_space])
                
                self.replay_memory.append((state, action, reward, next_state, done))
                
                if (len(self.replay_memory) >= self.batch_size):
                    random_sample = self._sample_from_replay_memory()
                    self._predict_next_state_and_fit(*random_sample)
                    
                    if self.epsilon > self.min_epsilon:
                        self.epsilon *= self.decay
                
                reward_episode += reward
                state = next_state
                
                if (done):
                    print("Episode = {}, Score = {}".format(episode, reward))
                    break


# In[ ]:


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    
    env.seed(21)
    np.random.seed(21)
    
    model = DQN(env, __DQN_CONST__)
    model.train(__NUM_EPISODES__)
    


# In[ ]:




