import requests
import pickle
import zstd
import json

from bot_utils import *

#### Fake Halite ####

class Position:

    def __init__(self, x, y):
        self.x = x
        self.y = y

class Cell:

    def __init__(self):
        self.ship = None
        self.structure = None
        self.is_occupied = False
        self.has_structure = False
        self.halite_amount = 0

class Ship:

    def __init__(self):
        self.owner = None
        self.halite_amount = 0
        self.position = None
        self.id = None

class Shipyard:

    def __init__(self):
        self.position = None
        self.owner = None

class GameMap:

    def __init__(self, width, height):
        self.height = height
        self.width = width
        self.cells = None

    def __getitem__(self, pos):
        x, y = pos.x, pos.y
        return self.cells[y][x]


class Player:

    def __init__(self, id_):
        self.halite_amount = 0
        self.id = id_
        self.ships = []
        self.shipyard = None

    def get_ships(self):
        return self.ships


class Game:

    def __init__(self):
        self.game_map = None
        self.my_id = None
        self.turn_number = 0
        self.me = None

### Util ###

def frame_to_cmds(frame, id_):

    cmds = []

    moves = frame['moves'].get(id_, [])

    for move in moves:
        if move['type'] == 'g':
            cmds.append('g')
        elif 'direction' in move:
            dir = move['direction']
            ship_id = move['id']
            if dir != 'o':
                cmds.append(f'm {ship_id} {dir}')

    return cmds


def get_winner_id(data):

    max_production = 0

    for player_stat_data in data['game_statistics']['player_statistics']:
        max_production = max(max_production, player_stat_data['final_production'])

    for player_stat_data in data['game_statistics']['player_statistics']:
        if player_stat_data['final_production'] == max_production:
            return str(player_stat_data['player_id'])


def get_base_map(data):

    map_data = data['production_map']
    width, height = map_data['width'], map_data['height']
    map = [[0] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            map[y][x] = map_data['grid'][y][x]['energy']

    return (width, height), map


def get_ships(frame):

    entities = frame['entities']

    ships = []

    for player_id in entities:

        for ship_id in entities[player_id]:

            ship = Ship()
            ship.id = ship_id
            ship.owner = player_id
            ship.halite_amount = entities[player_id][ship_id]['energy']
            ship.position = Position(entities[player_id][ship_id]['x'], entities[player_id][ship_id]['y'])
            ships.append(ship)

    return ships


def get_shipyards(data):

    shipyards = []

    for player_data in data['players']:

        loc = player_data['factory_location']

        shipyard = Shipyard()
        shipyard.position = Position(loc['x'], loc['y'])
        shipyard.owner = str(player_data['player_id'])
        shipyards.append(shipyard)

    return shipyards


def get_player(frame, shipyards, ships, id_):

    player = Player(id_)
    player.ships = [ship for ship in ships if ship.owner == id_]
    player.halite_amount = frame['energy'][id_]
    player.shipyard = [ shipyard for shipyard in shipyards if shipyard.owner == id_ ][0]

    return player


def build_map(frame, base_map, shipyards, ships):

    width, height = len(base_map[0]), len(base_map)

    cells = [ [None] * width for _ in range(height) ]

    for y in range(height):
        for x in range(width):
            cell = Cell()
            cell.halite_amount = base_map[y][x]
            cells[y][x] = cell

    game_map = GameMap(width, height)
    game_map.cells = cells

    for ship in ships:
        cell = game_map[ship.position]
        cell.is_occupied = True
        cell.ship = ship

    for shipyard in shipyards:
        cell = game_map[shipyard.position]
        cell.has_structure = True
        cell.structure = shipyard

    for cell_data in frame['cells']:
        pos = Position(cell_data['x'], cell_data['y'])
        cell = game_map[pos]
        cell.halite_amount = cell_data['production']

    return game_map


def build_game(turn, base_map, shipyards, frame, me_id):

    all_ships = get_ships(frame)
    game_map = build_map(frame, base_map, shipyards, all_ships)
    winner_player = get_player(frame, shipyards, all_ships, me_id)

    game = Game()
    game.game_map = game_map
    game.my_id = me_id
    game.turn_number = turn
    game.me = winner_player

    return game

### Main Function ###

def generate_data(replay_url):

    dataset = []

    req = requests.get(replay_url)
    data = zstd.loads(req.content)
    data = json.loads(data.decode())

    winner_id = get_winner_id(data)

    (width, height), base_map = get_base_map(data)
    shipyards = get_shipyards(data)

    for i, frame in enumerate(data['full_frames']):

        game = build_game(i, base_map, shipyards, frame, winner_id)
        winner_cmds = frame_to_cmds(frame, winner_id)

        game_mat, game_vec = game_to_matrix(game)
        cmds_mat = commands_to_matrix(game, winner_cmds)

        dataset.append((game_mat, game_vec, cmds_mat))

    os.makedirs('train', exist_ok=True)
    with open(f'train\\{replay_url.split("/")[-2]}.halite.pkl', 'wb') as f:
        pickle.dump(dataset, f)

    return data


if __name__ == "__main__":

    data = generate_data('https://api.2018.halite.io/v1/api/user/0/match/3264235/replay')
