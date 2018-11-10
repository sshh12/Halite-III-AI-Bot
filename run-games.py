
from collections import defaultdict
from tqdm import tqdm
import subprocess
import random
import os
import re

NUM_GAMES = 10

ARCHIVE_DIR = 'archive'
ARCHIVE_FILE = 'MyBot.py'

MAIN_DIR = 'active-sota'
MAIN_FILE = 'MyBot.py'

HALITE_BIN = os.path.join('bin', 'halite.exe')
REPLAY_DIR = 'replays/'
PYTHON = 'python'

def main():

    re_score = re.compile("'([\w-]+)', was rank (\d+) with (\d+) halite")

    bots = [os.path.join(ARCHIVE_DIR, fn, ARCHIVE_FILE) for fn in os.listdir(ARCHIVE_DIR)]

    ranks = defaultdict(list)
    scores = defaultdict(list)

    for n in tqdm(range(NUM_GAMES), desc='Halite Matches', unit='game'):

        size = random.choice([32, 40, 48, 56, 64])
        num_players = random.choice([2, 4])

        players = random.sample(bots, num_players - 1)
        players.append(os.path.join(MAIN_DIR, MAIN_FILE))
        random.shuffle(players)

        cmd = [
            HALITE_BIN,
            '--replay-directory ' + REPLAY_DIR,
            '--no-timeout',
            '-vvv',
            '--height ' + str(size),
            '--width ' + str(size)
        ] + list(map(lambda p: PYTHON + ' ' + p, players))

        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in proc.stdout.readlines():
            match = re_score.search(line.decode("utf-8"))
            if match:
                name = match.group(1)
                rank = int(match.group(2))
                score = int(match.group(3))
                ranks[name].append(rank)
                scores[name].append(score)

    names = list(ranks)
    names.sort(key=lambda n: sum(scores[n]) / len(scores[n]), reverse=True)

    for i, name in enumerate(names):
        scs = scores[name]
        avg = sum(scs) / len(scs)
        print(i + 1, name, round(avg), scs)

if __name__ == "__main__":
    main()
