from contextlib import contextmanager
import numpy as np
import logging
import sys
import os

from hlt.positionals import Position

np.set_printoptions(threshold=np.nan)

@contextmanager
def import_quietly():

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import warnings
    def warn(*args, **kwargs):
        pass
    warnings.warn = warn

    stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')

    yield None

    sys.stderr = stderr

def game_to_matrix(game):

    game_map = game.game_map

    map_mat = np.zeros((game_map.height, game_map.width, 5), np.float32)

    # 0 dropoffs/shipyards
    # 1 my ships
    # 2 enemy dropoffs/shipyards
    # 3 enemy ships
    # 4 halite

    for x in range(game_map.height):

        for y in range(game_map.width):

            pos = Position(x, y)
            cell = game_map[pos]

            if cell.is_occupied:
                if cell.ship.owner == game.my_id:
                    map_mat[y, x, 1] = 1
                else:
                    map_mat[y, x, 3] = 1

            if cell.has_structure:
                if cell.structure.owner == game.my_id:
                    map_mat[y, x, 0] = 1
                else:
                    map_mat[y, x, 2] = 1

            map_mat[y, x, 4] = cell.halite_amount

    # norm halite
    map_mat[:, :, 4] = map_mat[:, :, 4] / 500

    return map_mat

def commands_to_matrix(game, cmds):

    game_map = game.game_map
    ships = game.me.get_ships()
    shipyard_pos = game.me.shipyard.position

    cmd_mat = np.zeros((game_map.height, game_map.width, 6), np.float32)

    # 0 stay
    # 1 spawn/dropoff
    # 2 up
    # 3 down
    # 4 left
    # 5 right

    for ship in ships:

        move = None
        for cmd in cmds:
            if str(ship.id) in cmd and not cmd.endswith('o'):
                move = cmd
                break

        if move:
            _, _, dir = move.split(' ')
            i = {'n': 2, 's': 3, 'w': 4, 'e': 5}[dir]
        else:
            i = 0

        pos = ship.position
        cmd_mat[pos.y, pos.x, i] = 1

    if 'g' in cmds:
        cmd_mat[shipyard_pos.y, shipyard_pos.x, 1] = 1
    else:
        cmd_mat[shipyard_pos.y, shipyard_pos.x, 0] = 1

    empty_idx = np.where(cmd_mat[:, :] == [0]*6)
    cmd_mat[empty_idx[0], empty_idx[1], 0] = 1

    return cmd_mat

def matrix_to_cmds(game, matrix):

    game_map = game.game_map
    ships = game.me.get_ships()
    shipyard_pos = game.me.shipyard.position

    cmds = []

    # 0 stay
    # 1 spawn/dropoff
    # 2 up
    # 3 down
    # 4 left
    # 5 right

    for x in range(game_map.height):

        for y in range(game_map.width):

            vec = matrix[y, x]

            # make one hot

            if vec[0] == 1:
                continue
            elif vec[1] == 1:
                if x == shipyard_pos.x and y == shipyard_pos.y:
                    cmds.append('g')
                else:
                    pass # dropoff logic
            else:
                cur_ship = None
                for ship in ships:
                    pos = ship.position
                    if x == pos.x and y == pos.y:
                        cur_ship = ship
                        break
                if vec[2] == 1: cmds.append(f'm {ship.id} n')
                if vec[3] == 1: cmds.append(f'm {ship.id} s')
                if vec[4] == 1: cmds.append(f'm {ship.id} w')
                if vec[5] == 1: cmds.append(f'm {ship.id} e')

    return cmds

def roll_to_position(entity, mat, crop=4):

    size = mat.shape[1]

    dx = entity.position.x % size - size // 2
    dy = entity.position.y % size - size // 2

    m = np.roll(mat, -dx, axis=2)
    m = np.roll(mat, -dy, axis=1)

    if crop > 0:

        mid = size // 2

        return m[:, mid-crop:mid+crop, mid-crop:mid+crop]

    return m
