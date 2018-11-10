
from hlt.positionals import Direction
from hlt import constants
import hlt

import random
import logging

from ship_agent import ShipAgent, ship_agents
from shipyard_agent import ShipyardAgent
from bot_utils import *

from AlgoBot import play

game = hlt.Game()

ShipAgent.load_model()
ShipyardAgent.load_model()

game.ready("PPOAlgoBot")

logging.info("Successfully created bot! My Player ID is {}.".format(game.my_id))

shipyard_agent = ShipyardAgent(game)
shipyard_agent.update(game_to_matrix(game))

while True:

    game.update_frame()

    me = game.me
    game_map = game.game_map

    map_matrix = game_to_matrix(game)

    cmds = []

    #### SHIPS ####

    cur_ships = me.get_ships()

    logging.info('Ships: ' + str(len(cur_ships)))

    ShipAgent.process_dead(cur_ships)

    for ship in cur_ships:

        if ship.id not in ship_agents:
            ship_agents[ship.id] = ShipAgent(ship, game)

        agent = ship_agents[ship.id]
        agent.update(ship, map_matrix)
        cmds.extend(agent.get_actions())

    #### SHIPYARD ####

    shipyard_agent.update(map_matrix)
    cmds.extend(shipyard_agent.get_actions())

    #### DONE ####

    # algo_cmds = play(game)
    #
    # for c in algo_cmds:
    #
    #     if c == 'g' and c not in cmds:
    #         shipyard_agent.reword = -2
    #         logging.info('!! ' + c)
    #
    #     elif c.startswith('m') and c not in cmds:
    #         _, id, d = c.split(' ')
    #         id = int(id)
    #         ship_agents[id].reword = -1
    #         logging.info('!! ' + c)
    #
    # for c in cmds:
    #
    #     if c == 'g' and c not in algo_cmds:
    #         shipyard_agent.reword = -2
    #         logging.info('!! ' + c)
    #
    #     elif c.startswith('m') and c not in algo_cmds:
    #         _, id, d = c.split(' ')
    #         id = int(id)
    #         ship_agents[id].reword = -1
    #         logging.info('!! ' + c)

    logging.info(cmds)
    #logging.info(algo_cmds)

    game.end_turn(cmds)
