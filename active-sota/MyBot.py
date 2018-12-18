
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

import pickle
import os
import uuid
import random
import logging

logging.info("[ConvBot] Successfully created bot!")

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

        with open('test.txt', 'a') as e:
            e.write(str(cmds) + '\n')

        game.end_turn(cmds)

model_agent()
