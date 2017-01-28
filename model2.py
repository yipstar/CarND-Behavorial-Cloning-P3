# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.optimizers import Adam
from keras.models import model_from_json

import pandas as pd

from generator2 import load_data

refine = True
if refine:

    with open("model.json", 'r') as jfile:
        model = model_from_json(jfile.read())

    model.load_weights("model.h5")

    data_dir = "joystick_data2"
    driving_log = pd.read_csv(data_dir + "/driving_log.csv")
    X_train, y_train = load_data(driving_log)

    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)

    adam = Adam(lr=0.0001)
    num_epochs = 5

else:

    data_dir = "udacity_data"
    driving_log = pd.read_csv(data_dir + "/driving_log.csv")
    X_train, y_train = load_data(driving_log)

    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)

    from nvidia_model2 import NvidiaModel
    model = NvidiaModel()

    adam = Adam(lr=0.001)
    num_epochs = 10

model.compile(optimizer=adam, loss="mse")

history = model.fit(X_train, y_train, batch_size=128, nb_epoch=num_epochs, validation_split=0.05)

print("Saving model weights and configuration file.")
model.save_weights("model.h5")

with open('model.json', 'w') as outfile:
    outfile.write(model.to_json())
