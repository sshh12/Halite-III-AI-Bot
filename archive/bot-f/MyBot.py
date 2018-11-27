
from hlt.positionals import Direction
from hlt import constants
import hlt

import random
import logging

from ship_agent import ShipAgent, ship_agents
from shipyard_agent import ShipyardAgent
from bot_utils import *

# from AlgoBot2 import play

game = hlt.Game()

ShipAgent.load_model()
ShipyardAgent.load_model()

game.ready("PPOAlgoBot-F-Old")

logging.info("[MAIN] Successfully created bot! My Player ID is {}.".format(game.my_id))

shipyard_agent = ShipyardAgent(game)
shipyard_agent.update(game_to_matrix(game))

while True:

    game.update_frame()

    me = game.me
    game_map = game.game_map
    cur_ships = me.get_ships()
    map_matrix = game_to_matrix(game)

    cmds = []

    #### EXPERT ####

    # algo_cmds = play(game)

    #### SHIPS ####

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

    ## Adjusting reward based on similarity to expert
    #
    # algo_movements = {}
    # ppo_movements = {}
    #
    # if 'g' in algo_cmds and 'g' not in cmds:
    #     shipyard_agent.reword = -10
    #     logging.info('!! g')
    # elif 'g' not in algo_cmds and 'g' in cmds:
    #     shipyard_agent.reword = -1
    #     logging.info('!! g')
    #
    # for c in algo_cmds:
    #     if c.startswith('m') and c not in cmds and 'o' not in c:
    #         _, id, d = c.split(' ')
    #         algo_movements[id] = d
    #         logging.info('!! ' + c)
    #
    # for c in cmds:
    #     if c.startswith('m') and c not in algo_cmds and 'o' not in c:
    #         _, id, d = c.split(' ')
    #         ppo_movements[id] = d
    #         logging.info('!! ' + c)
    #
    # for ship_id in ppo_movements:
    #
    #     if ship_id not in algo_movements:
    #         ship_agents[int(ship_id)].reword = -5
    #     else:
    #         ship_agents[int(ship_id)].reword = -1
    #
    # for ship_id in algo_movements:
    #
    #     if ship_id not in ppo_movements:
    #         ship_agents[int(ship_id)].reword = -3
    #     else:
    #         ship_agents[int(ship_id)].reword = -1

    logging.info(cmds)
    # logging.info(algo_cmds)

    game.end_turn(cmds)
