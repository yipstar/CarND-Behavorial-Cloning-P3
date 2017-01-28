# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.models import model_from_json

with open("model.json", 'r') as jfile:
    model = model_from_json(jfile.read())

model.load_weights("model.h5")

print(model.summary())
