#### IMPORTS ####

import os
import logging
from collections import defaultdict

from hlt.positionals import Direction, Position
from hlt import constants
import hlt

from bot_utils import *

with import_quietly():

    import numpy as np
    from keras.models import Model, load_model
    from keras.layers import Input, Dense, Dropout, Conv2D
    from keras import backend as K
    from keras.optimizers import Adam

#### CONSTANTS ####

#### Agent ####

class HaliteAgent:

    def __init__(self):
        pass
