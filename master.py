"""
The main script that controls the pipeline
"""

### Loading custom functions

from modules.photo_module import img_preproc, img_read
from modules.utility_module import read_NN_model

### Loading the model that is used in production 

main_model = read_NN_model('main_model/model_specs.json', 
                           'main_model/model_weights.h5')

### Reading and preprocesing all the photos

all_photo = img_read('photo_raw')
d = [img_preproc(x, 28, 28) for x in all_photo]

### Predicting the image labels 