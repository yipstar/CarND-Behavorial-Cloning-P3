from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import ELU, Dropout, Lambda
from keras.layers.convolutional import Convolution2D

def CommaAIModel():
    model = Sequential()

    input_shape = (64, 64, 3)

    # Perform image normalizaton in a lambda layer
    model.add(Lambda(lambda x: x/255. - 0.5, input_shape = input_shape))

    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))

    # model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same",
    #                         input_shape=(160, 320, 3)))

    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    return model
