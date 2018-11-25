"""
The main script that controls the pipeline
"""

### Loading modules

import numpy as np
import pandas as pd

### Loading custom functions

from modules.photo_module import create_path_frame, img_read
from modules.utility_module import read_NN_model, construct_fit_frame

### Reading the class decoder

class_df = pd.read_csv('main_model/class_decoder.csv')

### Loading the model that is used in production 

main_model = read_NN_model('main_model/model_specs.json', 
                           'main_model/model_weights.h5')

### Reading and preprocesing all the photos

all_photo = create_path_frame('input', return_mapper = True) 
d = [img_read(x, h = 28, w = 28) for x in all_photo['path']]
d = np.asarray(d)

### Predicting the image label probabilities

fit = main_model.predict(d)

### Constructing a data frame to store the results in 

fit_df = construct_fit_frame(fit, class_df)
fit_df = fit_df.merge(all_photo, on = 'image_nr')
fit_df = fit_df.sort_values(['image_nr'], ascending = True)

### Saving the results

fit_df.to_csv('output/fitted_clases.csv', index = False)

