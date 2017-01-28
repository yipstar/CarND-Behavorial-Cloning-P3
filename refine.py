import pandas as pd

# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.optimizers import Adam
from keras.models import model_from_json

with open("model.json", 'r') as jfile:
    model = model_from_json(jfile.read())

model.load_weights("model.h5")

# refine the model with new training data

data_dir = "new_data2"
driving_log = pd.read_csv(data_dir + "/driving_log.csv")

adam = Adam(lr=0.00001)
model.compile(optimizer=adam, loss="mse")

# The probability of dropping data with small angles
pr_threshold = .5
batch_size = 16
EPOCHS = 1

from generator import data_generator

for epoch_index in range(EPOCHS):
    datagen = data_generator(driving_log, pr_threshold, batch_size)
    model.fit_generator(datagen, samples_per_epoch=2000, nb_epoch=1, verbose=1)
    # reduce the probability of dropping small angle data on each iteration
    pr_threshold = 1 / (epoch_index + 1) * 1
    print('Keep probability:' + str(pr_threshold))

print("Saving model weights and configuration file.")
model.save_weights("model.h5")

with open('model.json', 'w') as outfile:
    outfile.write(model.to_json())
