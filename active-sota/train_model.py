import os
import pickle
import numpy as np

from keras.models import Model
from keras.layers import Input, Dense
from keras.layers.convolutional import Conv2D

maps = []
vecs = []
actions = []

for fn in os.listdir('train'):

    with open(os.path.join('train', fn), 'rb') as f:
        data = pickle.load(f)
        for item in data:
            map, vec, action = item
            maps.append(map)
            vecs.append(vec)
            actions.append(action)

maps = np.array(maps)
vecs = np.array(vecs)
actions = np.array(actions)

indices = np.arange(maps.shape[0])
np.random.shuffle(indices)
maps = maps[indices]
vecs = vecs[indices]
actions = actions[indices]

print(maps.shape, vecs.shape, actions.shape)

map_input = Input(shape=(32, 32, 5))
x = Conv2D(32, kernel_size=4, activation='relu', padding='same')(map_input)
x = Conv2D(32, kernel_size=4, activation='relu', padding='same')(x)
x = Conv2D(6, kernel_size=4, activation='relu', padding='same')(x)
out = x

model = Model(inputs=map_input, outputs=out)
model.compile('adam', loss='mse')

model.fit(x=[maps], y=actions, epochs=20, batch_size=20)

model.save('conv-model.h5')
