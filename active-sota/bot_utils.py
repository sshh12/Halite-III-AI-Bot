from contextlib import contextmanager
import numpy as np
import logging
import random
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
    me = game.me
    ships = game.me.get_ships()

    map_mat = np.zeros((game_map.height, game_map.width, 7), np.float32)

    # 0 dropoffs/shipyards
    # 1 my ships
    # 2 enemy dropoffs/shipyards
    # 3 enemy ships
    # 4 halite
    # 5 ship halite amt
    # 6 ship is full

    for y in range(game_map.height):

        for x in range(game_map.width):

            pos = Position(x, y)
            cell = game_map[pos]

            if cell.is_occupied:
                if cell.ship.owner == game.my_id:
                    map_mat[y, x, 1] = 1
                else:
                    map_mat[y, x, 3] = 1
                map_mat[y, x, 5] = cell.ship.halite_amount

            if cell.has_structure:
                if cell.structure.owner == game.my_id:
                    map_mat[y, x, 0] = 1
                else:
                    map_mat[y, x, 2] = 1

            map_mat[y, x, 4] = cell.halite_amount
            map_mat[y, x, 6] = (1 if cell.halite_amount >= 1000 else 0)

    # norm halite
    map_mat[:, :, 4] = np.sqrt(map_mat[:, :, 4] / 100)
    map_mat[:, :, 5] = map_mat[:, :, 5] / 500

    # non-spacial game information
    binary_encode = lambda num, pad: list(map(int, bin(num)[2:].zfill(pad)))[-pad:]

    game_vec = binary_encode(game.turn_number, 9)
    game_vec.extend(binary_encode(len(ships), 6))
    game_vec.append(np.log(me.halite_amount + 10) / 8)
    game_vec.append(me.halite_amount == 0)
    game_vec.append(me.halite_amount >= 1000)
    game_vec.append(me.halite_amount >= 4000)
    game_vec = np.array(game_vec, dtype=np.float32)

    return map_mat, game_vec


def cmd_for_ship(ship, cmds):

    id_ = str(ship.id)

    for cmd in cmds:
        parts = cmd.split(' ')
        if parts[0] == 'c' and parts[1] == id_:
            return cmd
        elif parts[0] == 'm' and parts[1] == id_ and parts[2] != 'o':
            return cmd

    return None


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

        move = cmd_for_ship(ship, cmds)

        pos = ship.position

        if not move:
            continue

        if 'c' in move:
            cmd_mat[pos.y, pos.x, 1] = 1

        else:
            _, _, dir = move.split(' ')
            i = {'n': 2, 's': 3, 'w': 4, 'e': 5}[dir]
            cmd_mat[pos.y, pos.x, i] = 1

    if 'g' in cmds:
        cmd_mat[shipyard_pos.y, shipyard_pos.x] = [0, 1, 0, 0, 0, 0]

    # fill rest w/stay
    for y in range(game_map.height):
        for x in range(game_map.width):
            if np.count_nonzero(cmd_mat[y, x]) == 6:
                cmd_mat[y, x, 0] = 1

    return cmd_mat


def get_ship_at_pos(x, y, ships):

    for ship in ships:
        pos = ship.position
        if x == pos.x and y == pos.y:
            return ship

    return None


def matrix_to_cmds(game, matrix):

    game_map = game.game_map
    me = game.me
    ships = me.get_ships()
    shipyard_pos = game.me.shipyard.position

    cmds = []

    # 0 stay
    # 1 spawn/dropoff
    # 2 up
    # 3 down
    # 4 left
    # 5 right

    for y in range(game_map.height):

        for x in range(game_map.width):

            vec = matrix[y, x]
            m = np.argmax(vec)

            if m == 0:
                continue

            elif m == 1:
                if x == shipyard_pos.x and y == shipyard_pos.y:
                    cmds.append('g')
                else:
                    cur_ship = get_ship_at_pos(x, y, ships)
                    if not cur_ship:
                        continue
                    cmds.append(cur_ship.make_dropoff())

            elif m in [2, 3, 4, 5]:
                cur_ship = get_ship_at_pos(x, y, ships)
                if not cur_ship:
                    continue
                dir = {2: 'n', 3: 's', 4: 'w', 5: 'e'}[m]
                cmds.append(f'm {cur_ship.id} {dir}')

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
