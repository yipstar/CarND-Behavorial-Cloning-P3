import json
import pandas as pd
import os
from keras.models import model_from_json
from keras.optimizers import Adam

# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf

from nvidia_model2 import NvidiaModel

refine = False
if refine:

    driving_log = pd.read_csv("refine_data/driving_log.csv")

    learning_rate = 0.00001
    samples_per_epoch = 20000
    p_threshold = 1
    num_epochs = 3

    with open("model.json", 'r') as jfile:
        model = model_from_json(jfile.read())
        model.load_weights("model.h5")

else:
    driving_log = pd.read_csv("udacity_data/driving_log.csv")
    validation_driving_log = pd.read_csv("training_data/driving_log.csv")

    learning_rate = 0.001
    samples_per_epoch = 20000
    nb_val_samples = 2000
    p_threshold = 1
    num_epochs = 10

    model = NvidiaModel()

adam = Adam(lr=learning_rate)
model.compile(optimizer=adam, loss="mse")

batch_size = 64

run_version = 21

dir = "working_model_version_" + str(run_version)
if not os.path.exists(dir):
    os.makedirs(dir)

from generator2 import data_generator

for epoch_index in range(num_epochs):

    datagen = data_generator(driving_log, batch_size, p_threshold)
    validation_datagen = data_generator(validation_driving_log, batch_size, p_threshold)

    model.fit_generator(datagen, samples_per_epoch=samples_per_epoch, nb_epoch=1, verbose=1, validation_data=validation_datagen, nb_val_samples=nb_val_samples)

    # increase the probability of dropping small angle data on each iteration
    p_threshold = 1 / (epoch_index + 1) * 1
    print('Keep probability:' + str(p_threshold))

    dir = "working_model_version_" + str(run_version) + "/epoch_" + str(epoch_index)
    if not os.path.exists(dir):
        os.makedirs(dir)

    print("Saving model weights and configuration file.")
    model.save_weights(dir + "/model.h5")

    with open(dir + '/model.json', 'w') as outfile:
        outfile.write(model.to_json())



