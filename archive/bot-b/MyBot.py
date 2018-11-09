
from hlt.positionals import Direction
from hlt import constants
import hlt

import random
import logging

from ship_agent2 import ShipAgent, ship_agents
from shipyard_agent2 import ShipyardAgent

game = hlt.Game()

ShipAgent.load_model()
ShipyardAgent.load_model()

game.ready("PPOBot-B-Old")

logging.info("Successfully created bot! My Player ID is {}.".format(game.my_id))

shipyard_agent = ShipyardAgent(game)

while True:

    game.update_frame()

    me = game.me
    game_map = game.game_map

    command_queue = []

    #### SHIPS ####

    cur_ships = me.get_ships()

    logging.info('Ships: ' + str(len(cur_ships)))

    ShipAgent.process_dead(cur_ships)

    for ship in cur_ships:

        if ship.id not in ship_agents:
            ship_agents[ship.id] = ShipAgent(ship, game)

        agent = ship_agents[ship.id]
        agent.update(ship)

        command_queue.extend(agent.get_actions())

    #### SHIPYARD ####

    shipyard_agent.update()
    command_queue.extend(shipyard_agent.get_actions())

    #### DONE ####

    logging.info(command_queue)

    game.end_turn(command_queue)
