
from hlt.positionals import Direction, Position
from hlt import constants
import hlt

import numpy as np
import random
import logging

def halite_matrix(game):

    game_map = game.game_map
    map_mat = np.zeros((game_map.height, game_map.width), np.float32)

    for x in range(game_map.height):

        for y in range(game_map.width):

            pos = Position(x, y)
            cell = game_map[pos]

            map_mat[y, x] = cell.halite_amount

    return map_mat

def largest_indices(ary, n):
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

def navigate(ship, pos, game_map, halite_mat):

    if isinstance(pos, Position):
        x, y = pos.x, pos.y

    elif isinstance(pos, tuple):
        x, y = pos

    cur_pos = ship.position

    logging.info(str(cur_pos) + " -> " + str((x, y)))

    if cur_pos.x == x and cur_pos.y == y:
        d = (0, 0)

    elif cur_pos.x > x:
        d = Direction.West

    elif cur_pos.x < x:
        d = Direction.East

    elif cur_pos.y < y:
        d = Direction.South

    elif cur_pos.y > y:
        d = Direction.North

    dp = Position(d[0], d[1])
    next_p = cur_pos + dp

    if next_p in claimed_pos: # or game_map[next_p].is_occupied:
        claimed_pos.append(cur_pos)
        d = (0, 0)
    else:
        claimed_pos.append(next_p)

    return d

class Ship:
    target = None
    state = 'finding'

ticks = 0

ships = {}
targets = []
claimed_pos = []

def play(game):

    global ticks, ships, targets, claimed_pos

    me = game.me
    game_map = game.game_map
    shipyard = me.shipyard

    cur_ships = me.get_ships()
    cur_ships.sort(key=lambda s: s.id)

    halite_mat = halite_matrix(game)

    ys, xs = largest_indices(halite_mat, 30)
    claimed_pos = []

    cmds = []

    for i, ship in enumerate(cur_ships):
        if ship.id not in ships:
            ships[ship.id] = Ship()
            ships[ship.id].state = 'finding'

            for k in range(30):
                pos = Position(xs[k], ys[k])
                if pos not in targets and pos != shipyard.position:
                    ships[ship.id].target = pos
                    targets.append(pos)
                    break

    for i, ship in enumerate(cur_ships):

        ship_obj = ships[ship.id]

        logging.info(ship_obj.state)

        if ship.position == ship_obj.target:

            if ship.halite_amount >= hlt.constants.MAX_HALITE or game_map[ship.position].halite_amount <= 10:
                ship_obj.state = 'delivery'
                if ship_obj.target in targets:
                    targets.remove(ship_obj.target)
            else:
                ship_obj.state = 'collecting'

        elif ship.position == shipyard.position and ship_obj.state == 'delivery':

            ship_obj.state = 'finding'

            for k in range(30):
                pos = Position(xs[k], ys[k])
                if pos not in targets and pos != shipyard.position:
                    ship_obj.target = pos
                    targets.append(pos)
                    break

        if ship_obj.state == 'finding':
            d = navigate(ship, ship_obj.target, game_map, halite_mat)
            cmds.append(ship.move(d))

        elif ship_obj.state == 'delivery':
            d = navigate(ship, shipyard.position, game_map, halite_mat)
            cmds.append(ship.move(d))

    if ticks <= 4 and shipyard.position not in claimed_pos:
        cmds.append(shipyard.spawn())

    ticks += 1

    return cmds

if __name__ == "__main__":
    game = hlt.Game()
    game.ready("AlgoBot")
    logging.info("Successfully created bot! My Player ID is {}.".format(game.my_id))
    while True:
        game.update_frame()
        game.end_turn(play(game))
