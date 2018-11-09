#### IMPORTS ####

import os
import logging
from collections import defaultdict

from hlt.positionals import Direction, Position
from hlt import constants
import hlt

from bot_utils import import_quietly

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

with import_quietly():

    import numpy as np
    from keras.models import Model, load_model
    from keras.layers import Input, Dense, Dropout
    from keras import backend as K
    from keras.optimizers import Adam

#### CONSTANTS ####

TRAIN = False
LOSS_CLIPPING = 0.2
NOISE = 0.01
GAMMA = 0.99
NUM_ACTIONS = 2
NUM_STATE = 44
HIDDEN_SIZE = 256
ENTROPY_LOSS = 5 * 1e-3
LR = 1e-4
BATCH_SIZE = 128
EPOCHS = 25

BATCH_DATA = {
    'obs': defaultdict(list),
    'action': defaultdict(list),
    'pred': defaultdict(list),
    'reword': defaultdict(list),
    'size': 0
}

DUMMY_ACTION, DUMMY_VALUE = np.zeros((1, NUM_ACTIONS)), np.zeros((1, 1))

MODELS_FOLDER = 'bot-b\\models'
ACTOR_MODEL = os.path.join(MODELS_FOLDER, 'shipyard-actor.h5')
CRITIC_MODEL = os.path.join(MODELS_FOLDER, 'shipyard-critic.h5')

#### CODE ####

def ppo_loss(advantage, old_prediction):

    def loss(y_true, y_pred):

        prob = K.sum(y_true * y_pred)
        old_prob = K.sum(y_true * old_prediction)
        r = prob/(old_prob + 1e-10)

        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage)) + ENTROPY_LOSS * (prob * K.log(prob + 1e-10))

    return loss

class ShipyardAgent:

    def __init__(self, game):

        self.game = game

        self.last_move_valid = True
        self.i = 0
        self.update()

    def update(self):

        self.shipyard = self.game.me.shipyard
        self.map = self.game.game_map
        self.ships = self.game.me.get_ships()

        if self.i > 1 and TRAIN:

            self._update_reword()

            BATCH_DATA['obs'][self.shipyard.id].append(self.last_obs)
            BATCH_DATA['action'][self.shipyard.id].append(self.last_action)
            BATCH_DATA['pred'][self.shipyard.id].append(self.last_pred)
            BATCH_DATA['reword'][self.shipyard.id].append(self.reword)
            BATCH_DATA['size'] += 1

            if BATCH_DATA['size'] >= BATCH_SIZE:
                ShipyardAgent.train_on_batch()

        self._update_observation()

        self.i += 1

    def _update_reword(self):

        if not self.last_move_valid:
            self.reword = -1
        else:
            gain = self.game.me.halite_amount - self.last_halite_amount
            if gain >= 0:
                self.reword = gain
            else:
                self.reword = gain / 100
            logging.info('Shipyard Reword ' + str(self.reword))

    def _update_observation(self):

        other_players = [ self.game.players[player_id] for player_id in self.game.players if player_id != self.game.my_id ]
        average_halite = sum([ player.halite_amount for player in other_players ]) / len(other_players)

        logging.info('Average Halite ' + str(average_halite) + ', My Halite ' + str(self.game.me.halite_amount))

        cur_obs = [
            self.game.turn_number > 5,
            self.game.turn_number > 100,
            self.game.turn_number > 200,
            self.game.turn_number > 400,
            len(self.ships) == 0,
            len(self.ships) == 1,
            len(self.ships) > 2,
            len(self.ships) > 8,
            len(other_players) == 1,
            average_halite > self.game.me.halite_amount,
            self.map[self.shipyard].is_occupied,
            self.game.me.halite_amount > constants.SHIP_COST
        ]

        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                if dx != 0 and dy != 0:
                    new_pos = self.shipyard.position + Position(dx, dy)
                    cur_obs.append(self.map[new_pos].is_occupied)
                    cur_obs.append(self.map[new_pos].halite_amount / 600)

        self.obs = self.last_obs = np.array(cur_obs, dtype=np.float32)

        self.last_halite_amount = self.game.me.halite_amount

    def _can_spawn(self):
        return self.game.me.halite_amount > constants.SHIP_COST and not self.map[self.shipyard].is_occupied

    def _generate_action(self):

        pred = ShipyardAgent.actor_model.predict([[self.obs], DUMMY_VALUE, DUMMY_ACTION])
        action = np.random.choice(NUM_ACTIONS, p=np.nan_to_num(pred[0]))
        action_matrix = np.zeros(pred[0].shape)
        action_matrix[action] = 1

        self.last_action = action_matrix
        self.last_pred = pred

        return action, action_matrix, pred

    def get_actions(self):

        self.last_move_valid = True

        action = self._generate_action()[0]

        if action == 0:
            if not self._can_spawn():
                self.last_move_valid = False
                return []
            return [self.game.me.shipyard.spawn()]
        else:
            return []

    @staticmethod
    def load_model():

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
                      loss=[ppo_loss(advantage=advantage, old_prediction=old_prediction)])

        if os.path.isfile(ACTOR_MODEL):
            model.load_weights(ACTOR_MODEL)

        ShipyardAgent.actor_model = model

        state_input = Input(shape=(NUM_STATE,))
        x = Dense(HIDDEN_SIZE, activation='relu')(state_input)
        x = Dropout(0.5)(x)
        x = Dense(HIDDEN_SIZE, activation='relu')(x)
        x = Dropout(0.5)(x)

        out_value = Dense(1)(x)

        model = Model(inputs=[state_input], outputs=[out_value])
        model.compile(optimizer=Adam(lr=LR), loss='mse')

        if os.path.isfile(CRITIC_MODEL):
            model.load_weights(CRITIC_MODEL)

        ShipyardAgent.critic_model = model

    @staticmethod
    def train_on_batch():

        global BATCH_DATA

        num_examples = BATCH_DATA['size']
        agents = BATCH_DATA['obs'].keys()

        observations = []
        actions = []
        old_predictions = []
        rewords = []

        for agent_id in agents:

            observations.extend(BATCH_DATA['obs'][agent_id])
            actions.extend(BATCH_DATA['action'][agent_id])
            old_predictions.extend(BATCH_DATA['pred'][agent_id])

            reword = BATCH_DATA['reword'][agent_id]

            for j in range(len(reword) - 2, -1, -1):
                reword[j] += reword[j + 1] * GAMMA

            rewords.extend(reword)

        observations = np.array(observations)
        actions = np.array(actions)
        old_predictions = np.array(old_predictions).reshape((num_examples, NUM_ACTIONS))
        rewords = np.array(rewords)

        pred_values = ShipyardAgent.critic_model.predict(observations)
        advantage = np.array(rewords - pred_values[:, 0])
        advantage = advantage.reshape((num_examples, 1))

        for _ in range(EPOCHS):
            ShipyardAgent.actor_model.train_on_batch([observations, advantage, old_predictions], [actions])
            ShipyardAgent.critic_model.train_on_batch([observations], [rewords])

        BATCH_DATA['obs'].clear()
        BATCH_DATA['action'].clear()
        BATCH_DATA['pred'].clear()
        BATCH_DATA['reword'].clear()
        BATCH_DATA['size'] = 0

        os.makedirs(MODELS_FOLDER, exist_ok=True)
        ShipyardAgent.actor_model.save(ACTOR_MODEL)
        ShipyardAgent.critic_model.save(CRITIC_MODEL)
