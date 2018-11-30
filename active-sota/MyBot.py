
from hlt.positionals import Direction
from hlt import constants
import hlt

from bot_utils import *

with import_quietly():
    from keras.models import load_model

import pickle
import os
import uuid
import random
import logging

logging.info("[ConvBot] Successfully created bot!")

def a():

    from AlgoBot2 import play

    DATA = []
    DATANAME = str(uuid.uuid4())

    game = hlt.Game()
    game.ready("TrainConv")

    while True:

        game.update_frame()

        me = game.me
        game_map = game.game_map

        algo_cmds = play(game)
        game_mat, game_vec = game_to_matrix(game)
        cmds_mat = commands_to_matrix(game, algo_cmds)
        cmds = matrix_to_cmds(game, cmds_mat)

        logging.info(algo_cmds)
        logging.info(cmds)

        game.end_turn(algo_cmds)

        os.makedirs('train', exist_ok=True)
        DATA.append((game_mat, game_vec, cmds_mat))
        with open(f'train\\{DATANAME}.dat', 'wb') as f:
            pickle.dump(DATA, f)


def b():

    model = load_model('conv-model.h5')

    game = hlt.Game()
    game.ready("ModelConv")

    while True:

        game.update_frame()

        game_mat, game_vec = game_to_matrix(game)

        pred = model.predict([np.array([game_mat])])

        cmds = matrix_to_cmds(game, pred[0])

        with open('test.txt', 'a') as e:
            e.write(str(cmds) + '\n')

        game.end_turn(cmds)

b()
