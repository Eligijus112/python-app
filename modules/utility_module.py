"""
Methods to deal with data wrangling/saving/loading/etc
"""

import tensorflow as tf
from tensorflow.keras.models import model_from_json

def read_NN_model(path_specs, path_weights):
    json_file = open(path_specs, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(path_weights)
    loaded_model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return loaded_model