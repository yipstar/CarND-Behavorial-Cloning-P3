from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import ELU, Dropout, Lambda, SpatialDropout2D
from keras.layers.convolutional import Convolution2D, Cropping2D
import tensorflow as tf

def resize(image):
    import tensorflow as tf
    return tf.image.resize_images(image, (66, 200))

def NvidiaModel():
    model = Sequential()

    input_shape = (66, 200, 3)

    # # Crop
    # model.add(Cropping2D(cropping=((22, 0), (0, 0)), input_shape=(160, 320, 3)))

    # # Resize
    # model.add(Lambda(resize))

    # Normalize
    model.add(Lambda(lambda x: x/255. - 0.5, input_shape = input_shape))
    # model.add(Lambda(lambda x: x/255. - 0.5))

    model.add(Convolution2D(24, 31, 98, subsample=(5, 5), border_mode="same"))
    # model.add(SpatialDropout2D(.2))
    model.add(ELU())

    model.add(Convolution2D(36, 14, 47, subsample=(5, 5), border_mode="same"))
    # model.add(SpatialDropout2D(.2))
    model.add(ELU())

    model.add(Convolution2D(48, 5, 22, subsample=(3, 3), border_mode="same"))
    # model.add(SpatialDropout2D(.2))
    model.add(ELU())

    model.add(Convolution2D(64, 3, 20, subsample=(3, 3), border_mode="same"))
    # model.add(SpatialDropout2D(.2))
    model.add(ELU())

    model.add(Convolution2D(64, 1, 18, subsample=(3, 3), border_mode="same"))
    # model.add(SpatialDropout2D(.2))
    model.add(ELU())

    model.add(Flatten())
    model.add(ELU())

    model.add(Dense(1164))
    model.add(ELU())

    model.add(Dense(100))
    model.add(Dropout(.5))
    model.add(ELU())

    model.add(Dense(50))
    model.add(ELU())

    model.add(Dense(10))
    model.add(Dropout(.5))
    model.add(ELU())

    model.add(Dense(1))

    return model
