from PIL import Image
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd

# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf

# BATCH_SIZE = 3
BATCH_SIZE = 128

# EPOCHS = 2
# EPOCHS = 20
EPOCHS = 8

# VALID_SPLIT = 0.5
VALID_SPLIT = 0.10

# n_train = X_train.shape[0]
# print("Number of training examples =", n_train)

# n_classes = len(np.unique(y_train))
# print("Number of classes =", n_classes)

# print("X_train shape: ", X_train.shape)
# print("y_train shape: ", y_train.shape)

# from keras.preprocessing.image import ImageDataGenerator
# datagen = ImageDataGenerator(
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True)

val_size = 1
data_dir = "udacity_data"
driving_log = pd.read_csv(data_dir + "/driving_log.csv")

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import ELU, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization

model = Sequential()

inputShape=(64, 64, 3)

# Perform image normalizaton in a lambda layer
model.add(Lambda(lambda x: x/255. - 0.5, input_shape = inputShape))

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

# model.add(Flatten(input_shape=(160, 320, 3)))
# model.add(ELU())
# model.add(Dense(512))
# model.add(ELU())
# model.add(Dense(1))

adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss="mse")

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
# datagen.fit(X_train)

# model.fit_generator(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
#                     samples_per_epoch=len(X_train), nb_epoch=EPOCHS)


# The probability of dropping data with small angles
pr_threshold = 1
batch_size = 64

from generator import data_generator

# X_test, y_test = data_generator(driving_log)
# print(X_test.shape)

for epoch_index in range(EPOCHS):
    datagen = data_generator(driving_log, pr_threshold, batch_size)
    model.fit_generator(datagen, samples_per_epoch=20000, nb_epoch=1, verbose=1)
    # reduce the probability of dropping small angle data on each iteration
    pr_threshold = 1 / (epoch_index + 1) * 1
    print('Keep probability:' + str(pr_threshold))


# history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, validation_split=VALID_SPLIT)

# test_score = model.evaluate(X_test, y_test)
# print("model test accuracy: ", test_score)

# print(model.metrics_names)

print("Saving model weights and configuration file.")

print(model.to_json())

model.save_weights("model.h5")

with open('model.json', 'w') as outfile:
    outfile.write(model.to_json())

# from keras.models import model_from_json

# with open('model.json', 'r') as jfile:
#     model2 = model_from_json(jfile.read())
#     print(model2)
