
from hlt.positionals import Direction
from hlt import constants
import hlt

from bot_utils import *

with import_quietly():
    from keras.models import load_model
    import keras.losses
    def custom_xentropy(ytrue, ypred):
        # 0 stay
        # 1 spawn/dropoff
        # 2 up
        # 3 down
        # 4 left
        # 5 right
        xentropy = keras.losses.categorical_crossentropy(ytrue, ypred)
        xentropy = (1 - ytrue[:, :, 0]) * xentropy
        xentropy += 1 * ytrue[:, :, 1] * xentropy
        return xentropy
    keras.losses.custom_xentropy = custom_xentropy

import os
import random
import logging

logging.info("[ConvBot2] Successfully created bot!")

def collect_data():

    from ExpertBot import play
    import pickle
    import uuid

    DATA = []
    DATANAME = str(uuid.uuid4())

    game = hlt.Game()
    game.ready("TrainConv")

    i = 0

    while True:

        game.update_frame()

        me = game.me
        game_map = game.game_map

        game_mat, game_vec = game_to_matrix(game)

        algo_cmds = play(game)

        logging.info(algo_cmds)
        game.end_turn(algo_cmds)

        cmds_mat = commands_to_matrix(game, algo_cmds)
        DATA.append((game_mat, game_vec, cmds_mat))

        if i == 490 and me.halite_amount > 0:

            os.makedirs('train', exist_ok=True)
            with open(f'train\\{DATANAME}.halite.pkl', 'wb') as f:
                pickle.dump(DATA, f)

        i += 1

def model_agent():

    model = load_model('halite-conv-model.h5')

    game = hlt.Game()
    game.ready("ModelConv2")

    while True:

        game.update_frame()

        game_mat, game_vec = game_to_matrix(game)

        pred = model.predict([[game_mat], [game_vec]])

        actions = pred.reshape((1, 64, 64, 6))[0]

        cmds = matrix_to_cmds(game, actions)

        if 'g' in cmds and game.me.halite_amount < 1000:
            cmds.remove('g')

        with open('test.txt', 'a') as e:
            e.write(str(cmds) + '\n')

        game.end_turn(cmds)

model_agent()
