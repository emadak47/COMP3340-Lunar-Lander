import gym
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense 
from tensorflow.keras.activations import relu, linear 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error

from collections import deque
from constants import __DQN_CONST__


class DQN:
    def __init__(self, env, _para):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        
        self.lr = _para["LR"]
        self.gamma = _para["GAMMA"]
        self.epsilon = _para["EPSILON"]
        self.decay = _para["DECAY"]
        self.batch_size = _para["BATCH_SIZE"]
        self.min_epsilon = _para["MIN_EPSILON"]
        
        self.replay_memory = deque(maxlen=500000)
        self.model = self.create_model()
        
    def create_model(self):
        model = Sequential()
        model.add(Dense(512, input_dim=self.observation_space.shape[0], activation=relu))
        model.add(Dense(256, activation=relu))
        model.add(Dense(self.action_space.n, activation=linear))
        
        model.compile(loss=mean_squared_error,optimizer=Adam(lr=self.lr))
        print(model.summary())
        return model
     

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    
    env.seed(21)
    
    num_of_episodes = 1000
    
    model = DQN(env, __DQN_CONST__)
    