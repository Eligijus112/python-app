"""
Methods to deal with data wrangling/saving/loading/etc
"""

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import model_from_json

def read_NN_model(path_specs, path_weights,
                  loss_f = 'sparse_categorical_crossentropy', 
                  acc_metric = 'accuracy'):
    """
    Reads the specification and the weights from saved files. 
    Additionaly, compiles the model for instant use.
    """
    json_file = open(path_specs, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(path_weights)
    loaded_model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss = loss_f,
              metrics = [acc_metric])
    return loaded_model 
    
def extract_max_row(image_fit_df, max_column):
    """
    A very custom function that extracts the image label with the 
    highest probability. The probability is an outout of the neural network 
    model fit.
    """
    all_images = set(image_fit_df['image_nr'])
    all_images = list(all_images)
    
    results = []
    for img in all_images:
        subset = image_fit_df[image_fit_df['image_nr'] == img]
        subset = subset[subset[max_column] == max(subset[max_column])]
        results.append(subset.index.values)
    return image_fit_df[image_fit_df.index.isin(results)]       

def construct_fit_frame(fit, decoder_frame):
    """
    Construcs a frame that augments the information from the fited keras model
    """
    index = range(1, len(fit)+1)
    fit_df = pd.DataFrame(fit, index=index)
    fit_df['image_nr'] = index
    fit_df = fit_df.melt(id_vars = 'image_nr', var_name = 'class_code', 
                         value_name = 'p')
    fit_df = extract_max_row(fit_df, 'p')
    fit_df['class_code'] = fit_df.class_code.astype(int)
    decoder_frame['class_code'] = decoder_frame.class_code.astype(int)
    fit_df = fit_df.merge(decoder_frame, on = 'class_code')
    return fit_df