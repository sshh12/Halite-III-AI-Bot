#### IMPORTS ####

import os
import logging
from collections import defaultdict

from hlt.positionals import Direction, Position
from hlt import constants
import hlt

from bot_utils import *

with import_quietly():

    import numpy as np
    from keras.models import Model, load_model
    from keras.layers import Input, Dense, Dropout, Conv2D
    from keras import backend as K
    from keras.optimizers import Adam

#### CONSTANTS ####

TRAIN = True
LOSS_CLIPPING = 0.2
NOISE = 0.01
GAMMA = 0.99
NUM_ACTIONS = 7
NUM_STATE = 328
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

MODELS_FOLDER = 'models'
ACTOR_MODEL = os.path.join(MODELS_FOLDER, 'ship-actor.h5')
CRITIC_MODEL = os.path.join(MODELS_FOLDER, 'ship-critic.h5')

#### CODE ####

ship_agents = {}

def ppo_loss(advantage, old_prediction):

    def loss(y_true, y_pred):

        prob = K.sum(y_true * y_pred)
        old_prob = K.sum(y_true * old_prediction)
        r = prob/(old_prob + 1e-10)

        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage)) + ENTROPY_LOSS * (prob * K.log(prob + 1e-10))

    return loss

class ShipAgent:

    def __init__(self, ship, game):

        self.game = game

        self.last_move_valid = True
        self.i = 0

    def update(self, ship, mat, dead=False):

        self.map = self.game.game_map
        self.ship = ship

        if self.i > 0 and TRAIN:

            self._update_reword(dead)

            BATCH_DATA['obs'][ship.id].append(self.last_obs)
            BATCH_DATA['action'][ship.id].append(self.last_action)
            BATCH_DATA['pred'][ship.id].append(self.last_pred)
            BATCH_DATA['reword'][ship.id].append(self.reword)
            BATCH_DATA['size'] += 1

            if BATCH_DATA['size'] >= BATCH_SIZE:
                ShipAgent.train_on_batch()

        if not dead:

            self.mat = mat
            self._update_observation()
            self.i += 1

    def _update_reword(self, is_dead):

        if is_dead:
            if self._on_dropoff():
                self.reword = -1
            else:
                self.reword = -1 * min(2, self.last_halite_amount / 10)
                logging.info('Reword for death: ' + str(self.reword))
        elif not self.last_move_valid:
            self.reword = -1
        elif self._on_dropoff() and self.last_halite_amount > 0:
            self.reword = max(0, self.last_halite_amount - self.ship.halite_amount) * 10
            logging.info('Reword for deposit: ' + str(self.reword))
        else:
            self.reword = max(-1, self.ship.halite_amount - self.last_halite_amount) / 10
            logging.info('Reword for collection: ' + str(self.reword))

    def _on_dropoff(self):
        return self.map[self.ship.position].has_structure and self.map[self.ship.position].structure.owner == self.ship.owner

    def _update_observation(self):

        rel_mat = roll_to_position(self.ship, self.mat, crop=4)
        rel_mat = np.ndarray.flatten(rel_mat)

        home_dir = self.map.naive_navigate(self.ship, self.game.me.shipyard.position)

        game_state = [
            self.ship.is_full,
            self.map[self.ship.position].has_structure,
            self.map[self.ship.position].halite_amount / constants.MOVE_COST_RATIO <= self.ship.halite_amount,
            self._can_move(1),
            home_dir == Direction.North,
            home_dir == Direction.South,
            home_dir == Direction.East,
            home_dir == Direction.West
        ]

        game_states = np.array(game_state, dtype=np.float32)

        self.obs = self.last_obs = np.concatenate((rel_mat, game_states))

        self.last_halite_amount = self.ship.halite_amount

    def _can_move(self, move=0):

        if move == 0:
            return True

        return self.map[self.ship.position].halite_amount / constants.MOVE_COST_RATIO <= self.ship.halite_amount

    def _can_dropoff(self):
        return not self.map[self.ship.position].has_structure and self.game.me.halite_amount > constants.DROPOFF_COST

    def _generate_action(self):

        pred = ShipAgent.actor_model.predict([[self.obs], DUMMY_VALUE, DUMMY_ACTION])
        action = pick_action(pred[0], temp=1.5)
        action_matrix = np.zeros(pred[0].shape)
        action_matrix[action] = 1

        self.last_action = action_matrix
        self.last_pred = pred

        return action, action_matrix, pred

    def get_actions(self):

        self.last_move_valid = True

        action = self._generate_action()[0]

        if 0 <= action <= 4:
            if not self._can_move(action):
                self.last_move_valid = False
                return []
            return [self.ship.move('onsew'[action])]
        elif action == 5:
            if not self._can_dropoff():
                self.last_move_valid = False
                return []
            return [self.ship.make_dropoff()]
        else:
            return []

    @staticmethod
    def process_dead(alive_ships):

        alive_ship_ids = { ship.id for ship in alive_ships }
        all_ship_ids = set(ship_agents)

        for ship_id in all_ship_ids:

            if ship_id not in alive_ship_ids:

                logging.info(str(ship_id) + ' is dead')

                ship_agents[ship_id].update(ship_agents[ship_id].ship, None, dead=True)

                del ship_agents[ship_id]

    @staticmethod
    def load_model():

        state_input = Input(shape=(NUM_STATE,))
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(NUM_ACTIONS,))

        x = Dense(256, activation='relu')(state_input)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)

        out_actions = Dense(NUM_ACTIONS, activation='softmax', name='output')(x)

        model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
        model.compile(optimizer=Adam(lr=LR),
                      loss=[ppo_loss(advantage=advantage, old_prediction=old_prediction)])

        if os.path.isfile(ACTOR_MODEL):
            model.load_weights(ACTOR_MODEL)

        ShipAgent.actor_model = model

        state_input = Input(shape=(NUM_STATE,))
        x = Dense(256, activation='relu')(state_input)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)

        out_value = Dense(1)(x)

        model = Model(inputs=[state_input], outputs=[out_value])
        model.compile(optimizer=Adam(lr=LR), loss='mse')

        if os.path.isfile(CRITIC_MODEL):
            model.load_weights(CRITIC_MODEL)

        ShipAgent.critic_model = model

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

        pred_values = ShipAgent.critic_model.predict(observations)
        advantage = np.array(rewords - pred_values[:, 0])
        advantage = advantage.reshape((num_examples, 1))

        for _ in range(EPOCHS):
            ShipAgent.actor_model.train_on_batch([observations, advantage, old_predictions], [actions])
            ShipAgent.critic_model.train_on_batch([observations], [rewords])

        BATCH_DATA['obs'].clear()
        BATCH_DATA['action'].clear()
        BATCH_DATA['pred'].clear()
        BATCH_DATA['reword'].clear()
        BATCH_DATA['size'] = 0

        os.makedirs(MODELS_FOLDER, exist_ok=True)
        ShipAgent.actor_model.save(ACTOR_MODEL)
        ShipAgent.critic_model.save(CRITIC_MODEL)
