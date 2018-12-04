import os
import pickle
import numpy as np

from keras.models import Model
from keras.layers import *
from keras.losses import *
from keras.metrics import *
from keras import backend as K
from keras.utils import plot_model

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
actions = actions.reshape((actions.shape[0], 1024, 6))

indices = np.arange(maps.shape[0])
np.random.shuffle(indices)
maps = maps[indices]
vecs = vecs[indices]
actions = actions[indices]

print(maps.shape, vecs.shape, actions.shape)

map_input = Input(shape=(32, 32, 5))
game_vec_input = Input(shape=(12,))

x = Conv2D(16, kernel_size=3, activation='selu', padding='same', kernel_initializer='he_normal')(map_input)
x = Conv2D(16, kernel_size=3, activation='selu', padding='same', kernel_initializer='he_normal')(x)

x = b = Dropout(0.25)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(32, kernel_size=3, activation='selu', padding='same', kernel_initializer='he_normal')(x)
x = Conv2D(32, kernel_size=3, activation='selu', padding='same', kernel_initializer='he_normal')(x)

x = a = Dropout(0.25)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(64, kernel_size=3, activation='selu', padding='same', kernel_initializer='he_normal')(x)

game_vec_reshaped = RepeatVector(8 * 8)(game_vec_input)
game_vec_reshaped = Reshape((8, 8, int(game_vec_input.shape[1])))(game_vec_reshaped)

x = concatenate([x, game_vec_reshaped], axis=3)

x = Conv2D(64, kernel_size=3, activation='selu', padding='same', kernel_initializer='he_normal')(x)

x = Dropout(0.25)(x)
x = Conv2DTranspose(32, (2, 2), strides=(2, 2))(x)

x = Conv2D(32, kernel_size=3, activation='selu', padding='same', kernel_initializer='he_normal')(x)
x = concatenate([x, a], axis=3)
x = Conv2D(32, kernel_size=3, activation='selu', padding='same', kernel_initializer='he_normal')(x)

x = Conv2DTranspose(16, (2, 2), strides=(2, 2))(x)

x = Conv2D(16, kernel_size=3, activation='selu', padding='same', kernel_initializer='he_normal')(x)
x = concatenate([x, b], axis=3)
x = Conv2D(12, kernel_size=3, activation='selu', padding='same', kernel_initializer='he_normal')(x)
x = Conv2D(6, kernel_size=1, activation='linear', padding='same', kernel_initializer='he_normal')(x)
x = Reshape((6, 32 * 32))(x)
x = Permute((2, 1))(x)
x = Activation('softmax')(x)

out = x

model = Model(inputs=[map_input, game_vec_input], outputs=out)

def ignore_unknown_xentropy(ytrue, ypred):
    return (1 - ytrue[:, :, 0]) * categorical_crossentropy(ytrue, ypred)

model.compile('adam', loss=ignore_unknown_xentropy, metrics=['acc'])

if __name__ == '__main__':

    model.fit(x=[maps, vecs], y=actions, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

    model.save('conv-model.h5')
    plot_model(model, to_file='model.png')
