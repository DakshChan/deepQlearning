# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import huber_loss
from keras import backend as K
EPISODES = 5000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.1 # discount rate
        self.gamma_max = 0.995
        self.gamma_rise = 1.005
        self.epsilon = 1.2  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    """
    Huber loss for Q Learning

    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                # a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if self.gamma < self.gamma_max:
            self.gamma *= self.gamma_rise
            if self.gamma > self.gamma_max:
                self.gamma = self.gamma_max
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save(name)


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    print(state_size)
    action_size = env.action_space.n
    print(action_size)
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-ddqn-EP20.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        time = 0  # time is used just to count frames as a measurement of how long the ai lasted
        while True:
            # env.render()
            time += 1
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            # for this problem we want the pole to balance forever
            # so giving a negative reward if its finished should train the ai to play forever
            if done:
                reward = -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model()
                print("episode: {}/{}, score: {}, e: {:.2}, g: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon, agent.gamma))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        #this saves every 20 episodes
        if e % 20 == 0:
            agent.save("./save/cartpole-ddqn-EP"+str(e)+".h5")