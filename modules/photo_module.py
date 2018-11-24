"""
The methods in this script will deal with images
"""

import cv2
import os
import re
import pandas as pd

def img_read(path, append_path = True, return_mapper = False):
    all_photo = os.listdir(path)
    good_img = []
    for f in all_photo:
        if append_path: 
            f = path + '/' + f
        if re.search(r'.jpg$|.png$|.jpeg$', f) is not None:
            good_img.append(f)
    
    result = pd.DataFrame(good_img)
    result = result.rename({0 : 'path'}, axis = 'columns')
    
    if(return_mapper):        
        result['image_nr'] = range(1, len(good_img)+1)      
        result = result.rename({0 : 'image_nr'}, axis = 'columns')
    
    return result
    
def img_preproc(path, h, w, to_grey = True):
    img = cv2.imread(path)
    img = cv2.resize(img, (h, w))
    if(to_grey):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255    
    return img   
