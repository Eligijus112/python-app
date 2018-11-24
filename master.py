"""
The main script that controls the pipeline
"""

### Loading custom functions

from modules.photo_module import img_preproc, img_read

### Reading and preprocesing all the photos

all_photo = img_read('photo_raw')
d = [img_preproc(x, 28, 28) for x in all_photo]

