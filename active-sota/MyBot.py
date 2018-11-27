
from hlt.positionals import Direction
from hlt import constants
import hlt

import random
import logging

from halite_agent import HaliteAgent
from bot_utils import *
import pickle
import os
import uuid

from AlgoBot2 import play

game = hlt.Game()

game.ready("ConvBot")

logging.info("[ConvBot] Successfully created bot! My Player ID is {}.".format(game.my_id))

DATA = []
DATANAME = str(uuid.uuid4())

while True:

    game.update_frame()

    me = game.me
    game_map = game.game_map

    algo_cmds = play(game)
    game_mat = game_to_matrix(game)
    cmds_mat = commands_to_matrix(game, algo_cmds)
    cmds = matrix_to_cmds(game, cmds_mat)

    logging.info(algo_cmds)
    logging.info(cmds)

    game.end_turn(algo_cmds)

    os.makedirs('train', exist_ok=True)
    DATA.append((game_mat, cmds_mat))
    with open(f'train\\{DATANAME}.dat', 'wb') as f:
        pickle.dump(DATA, f)
