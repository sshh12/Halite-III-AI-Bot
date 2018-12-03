import os
import pickle
import numpy as np

from keras.models import Model
from keras.layers import *
from keras import backend as K

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

def depth_softmax(matrix):
    sigmoid = lambda x: 1 / (1 + K.exp(-x))
    sigmoided_matrix = sigmoid(matrix)
    softmax_matrix = sigmoided_matrix / K.sum(sigmoided_matrix, axis=0)
    return softmax_matrix

print(maps.shape, vecs.shape, actions.shape)

map_input = Input(shape=(32, 32, 5))

x = Conv2D(16, kernel_size=3, activation='selu', padding='same', kernel_initializer='he_normal')(map_input)
x = Conv2D(16, kernel_size=3, activation='selu', padding='same', kernel_initializer='he_normal')(x)

x = b = Dropout(0.5)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(32, kernel_size=3, activation='selu', padding='same', kernel_initializer='he_normal')(x)
x = Conv2D(32, kernel_size=3, activation='selu', padding='same', kernel_initializer='he_normal')(x)

x = a = Dropout(0.5)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(64, kernel_size=3, activation='selu', padding='same', kernel_initializer='he_normal')(x)
x = Conv2D(64, kernel_size=3, activation='selu', padding='same', kernel_initializer='he_normal')(x)

x = Dropout(0.5)(x)
x = UpSampling2D(size=(2, 2))(x)

x = Conv2D(32, kernel_size=3, activation='selu', padding='same', kernel_initializer='he_normal')(x)
x = concatenate([x, a], axis=3)
x = Conv2D(32, kernel_size=3, activation='selu', padding='same', kernel_initializer='he_normal')(x)

x = UpSampling2D(size=(2, 2))(x)

x = Conv2D(16, kernel_size=3, activation='selu', padding='same', kernel_initializer='he_normal')(x)
x = concatenate([x, b], axis=3)
x = Conv2D(12, kernel_size=3, activation='selu', padding='same', kernel_initializer='he_normal')(x)
x = Conv2D(6, kernel_size=1, activation='linear', padding='same', kernel_initializer='he_normal')(x)
x = Lambda(depth_softmax)(x)

out = x

model = Model(inputs=map_input, outputs=out)
model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x=[maps], y=actions, epochs=5, batch_size=32, validation_split=0.1)

model.save('conv-model.h5')
