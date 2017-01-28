from PIL import Image
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd

# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.optimizers import Adam

# from comma_model import CommaAIModel
# model = CommaAIModel()

from nvidia_model import NvidiaModel
model = NvidiaModel()

adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss="mse")

# The probability of dropping data with small angles
pr_threshold = 1
batch_size = 32
EPOCHS = 20

from generator import data_generator

data_dir = "udacity_data"
driving_log = pd.read_csv(data_dir + "/driving_log.csv")

for epoch_index in range(EPOCHS):
    datagen = data_generator(driving_log, pr_threshold, batch_size)
    model.fit_generator(datagen, samples_per_epoch=20000, nb_epoch=1, verbose=1)

    # reduce the probability of dropping small angle data on each iteration
    pr_threshold = 1 / (epoch_index + 1) * 1
    print('Drop straight steering probability:' + str(pr_threshold))

print("Saving model weights and configuration file.")
model.save_weights("model.h5")

with open('model.json', 'w') as outfile:
    outfile.write(model.to_json())
