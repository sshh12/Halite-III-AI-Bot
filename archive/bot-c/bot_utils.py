from contextlib import contextmanager
import numpy as np
import logging
import sys
import os

from hlt.positionals import Position

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

def pick_action(pvals, temp=1.0):

    preds = np.nan_to_num(pvals)
    preds = np.log(preds + 1e-8) / temp
    preds = np.exp(preds)
    preds = preds / np.sum(preds)

    pred = np.random.choice(pvals.shape[0], p=preds)

    return pred

def game_to_matrix(game):

    game_map = game.game_map

    map_mat = np.zeros((5, game_map.height, game_map.width), np.float32)

    # dropoffs
    # my ships
    # enemy dropoffs
    # enemy ships
    # halite

    for x in range(game_map.height):

        for y in range(game_map.width):

            pos = Position(x, y)
            cell = game_map[pos]

            if cell.is_occupied:
                if cell.ship.owner == game.my_id:
                    map_mat[1, y, x] = 1
                else:
                    map_mat[3, y, x] = 1

            if cell.has_structure:
                if cell.structure.owner == game.my_id:
                    map_mat[0, y, x] = 1
                else:
                    map_mat[2, y, x] = 1

            map_mat[4, y, x] = cell.halite_amount

    # average halite
    map_mat[4, :, :] = map_mat[4, :, :] / np.mean(map_mat[4, :, :])

    return map_mat

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
