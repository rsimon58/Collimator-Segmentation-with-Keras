from keras.layers import Input
from keras.models import Model
from keras.layers import Activation
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras import backend as K
from layers import RectInterpolation
import numpy as np

#def get_initial_weights(output_size):
#    b = np.zeros((2, 3), dtype='float32')
#    b[0, 0] = 1
#    b[1, 1] = 1
#    W = np.zeros((output_size, 6), dtype='float32')
#    weights = [W, b.flatten()]
#    return weights

def get_initial_weights(output_size):
    b = np.zeros((5), dtype='float32')
    b[1] = 1
    b[2] = 1
    W = np.zeros((output_size, 5), dtype='float32')
    weights = [W, b]
    return weights


def make_square_images(batch_size,rows,cols, width, height):
    image = np.zeros((rows,cols), dtype='float32')
    image[43:53,35:44] = 1.0
    image = image.flatten()
    image = np.tile(image, np.stack([batch_size]))
    image = np.reshape(image, (batch_size, rows, cols))
    image = np.expand_dims(image, 3)
    return image

def create_RecNet(rows, cols):
    input_size = (rows, cols, 1)
    image = Input(shape=input_size)

    #locnet = Conv2D(20, (3, 3))(image)
    #locnet = MaxPool2D(pool_size=(2, 2))(locnet)
    #locnet = Conv2D(20, (5, 5))(locnet)
    #locnet = Flatten()(locnet)
    #locnet = Dense(50)(locnet)
    #locnet = Activation('relu')(locnet)
    #weights = get_initial_weights(50)
    #locnet = Dense(5, weights=weights)(locnet)

    locnet = Conv2D(16, (5, 5), padding='same', activation='relu')(image)
    locnet = Conv2D(32, (5, 5), padding='same', activation='relu')(locnet)
    locnet = MaxPool2D(pool_size=(2, 2))(locnet)
    locnet = Flatten()(locnet)
    locnet = Dense(64)(locnet)
    weights = get_initial_weights(64)
    locnet = Dense(5, weights=weights)(locnet)

    sampling_size = (rows,cols)

    rectImage = make_square_images(32, rows, cols, 10, 10)

    x = RectInterpolation(sampling_size, rectImage)([image, locnet])

    return Model(inputs=[image], outputs=[x])

