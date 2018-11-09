import numpy as np

import gym

from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras import backend as K
from keras.optimizers import Adam

import numba as nb
from tensorboardX import SummaryWriter

EPISODES = 1000000

LOSS_CLIPPING = 0.2
EPOCHS = 10
NOISE = 0.01

GAMMA = 0.99

BATCH_SIZE = 256
NUM_ACTIONS = 4
NUM_STATE = 24
HIDDEN_SIZE = 256
ENTROPY_LOSS = 5 * 1e-3
LR = 1e-4

DUMMY_ACTION, DUMMY_VALUE = np.zeros((1, NUM_ACTIONS)), np.zeros((1, 1))


def exponential_average(old, new, b1):
    return old * b1 + (1-b1) * new


def proximal_policy_optimization_loss(advantage, old_prediction):

    def loss(y_true, y_pred):
        prob = K.sum(y_true * y_pred)
        old_prob = K.sum(y_true * old_prediction)
        r = prob/(old_prob + 1e-10)

        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage)) + ENTROPY_LOSS * (prob * K.log(prob + 1e-10))

    return loss

class Agent:
    def __init__(self):
        self.critic = self.build_critic()
        self.actor = self.build_actor()
        self.episode = 0
        self.observation = self.env.reset()
        self.reward = []
        self.reward_over_time = []
        self.gradient_steps = 0

    def build_actor(self):

        state_input = Input(shape=(NUM_STATE,))
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(NUM_ACTIONS,))

        x = Dense(HIDDEN_SIZE, activation='relu')(state_input)
        x = Dropout(0.5)(x)
        x = Dense(HIDDEN_SIZE, activation='relu')(x)
        x = Dropout(0.5)(x)

        out_actions = Dense(NUM_ACTIONS, activation='softmax', name='output')(x)

        model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
        model.compile(optimizer=Adam(lr=LR),
                      loss=[proximal_policy_optimization_loss(
                          advantage=advantage,
                          old_prediction=old_prediction)])
        model.summary()

        return model

    def build_critic(self):

        state_input = Input(shape=(NUM_STATE,))
        x = Dense(HIDDEN_SIZE, activation='relu')(state_input)
        x = Dropout(0.5)(x)
        x = Dense(HIDDEN_SIZE, activation='relu')(x)
        x = Dropout(0.5)(x)

        out_value = Dense(1)(x)

        model = Model(inputs=[state_input], outputs=[out_value])
        model.compile(optimizer=Adam(lr=LR), loss='mse')

        return model

    def reset_env(self):
        self.episode += 1
        self.observation = self.env.reset()
        self.reward = []

    def get_action(self):
        p = self.actor.predict([self.observation.reshape(1, NUM_STATE), DUMMY_VALUE, DUMMY_ACTION])
        action = np.random.choice(NUM_ACTIONS, p=np.nan_to_num(p[0]))
        action_matrix = np.zeros(p[0].shape)
        action_matrix[action] = 1
        return action, action_matrix, p

    def get_action_continuous(self):
        p = self.actor.predict([self.observation.reshape(1, NUM_STATE), DUMMY_VALUE, DUMMY_ACTION])
        action = action_matrix = p[0] + np.random.normal(loc=0, scale=NOISE, size=p[0].shape)
        return action, action_matrix, p

    def transform_reward(self):
        if self.episode % 100 == 0:
            print('Episode #', self.episode, '\tfinished with reward', np.array(self.reward).sum(),
                  '\tAverage reward of last 100 episode :', np.mean(self.reward_over_time[-100:]))
        self.reward_over_time.append(np.array(self.reward).sum())

        for j in range(len(self.reward) -1, -1, -1):
            self.reward[j] += self.reward[j + 1] * GAMMA

    def get_batch(self):
        batch = [[], [], [], []]

        tmp_batch = [[], [], []]
        while len(batch[0]) < BATCH_SIZE:
            action, action_matrix, predicted_action = self.get_action()
            observation, reward, done, info = self.env.step(action)
            self.reward.append(reward)

            tmp_batch[0].append(self.observation)
            tmp_batch[1].append(action_matrix)
            tmp_batch[2].append(predicted_action)
            self.observation = observation

            if done:
                self.transform_reward()
                for i in range(len(tmp_batch[0])):
                    obs, action, pred = tmp_batch[0][i], tmp_batch[1][i], tmp_batch[2][i]
                    r = self.reward[i]
                    batch[0].append(obs)
                    batch[1].append(action)
                    batch[2].append(pred)
                    batch[3].append(r)
                tmp_batch = [[], [], []]
                self.reset_env()

        obs, action, pred, reward = np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), np.reshape(np.array(batch[3]), (len(batch[3]), 1))
        pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
        return obs, action, pred, reward

    def run(self):
        while self.episode < EPISODES:
            obs, action, pred, reward = self.get_batch()
            old_prediction = pred
            pred_values = self.critic.predict(obs)

            advantage = reward - pred_values

            actor_loss = []
            critic_loss = []
            for e in range(EPOCHS):
                actor_loss.append(self.actor.train_on_batch([obs, advantage, old_prediction], [action]))
                critic_loss.append(self.critic.train_on_batch([obs], [reward]))

            self.gradient_steps += 1
