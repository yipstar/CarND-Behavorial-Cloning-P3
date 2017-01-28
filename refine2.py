import pandas as pd

# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.optimizers import Adam
from keras.models import model_from_json

from load_images import load_data

with open("model.json", 'r') as jfile:
    model = model_from_json(jfile.read())

model.load_weights("model.h5")

# refine the model with new training data

data_dir = "new_data"
X_train, y_train = load_data(data_dir)
print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)

adam = Adam(lr=0.00001)
model.compile(optimizer=adam, loss="mse")

history = model.fit(X_train, y_train, batch_size=64, nb_epoch=10, validation_split=0.05)

print("Saving model weights and configuration file.")
model.save_weights("model.h5")

with open('model.json', 'w') as outfile:
    outfile.write(model.to_json())
