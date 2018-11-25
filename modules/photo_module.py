"""
Methods that read and preproces the images
"""

import cv2
import os
import re
import pandas as pd

def create_path_frame(path, append_path = True, return_mapper = False):
    """
    Creates a dataframe with the links to the images in the *path* folder.  
    
    path (str): path to the folder where the images are
    append_path (bool): should we add the full path to the image?
    return_mapper (bool): should we return the image number?
    """
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
    

def img_read(path, h, w, to_grey = True):
    """
    Reads and preproces an image in *path* 
    
    h (float): height of the resized image
    w (float): width of the resized image.
    to_grey (bool): should the image be grey scale?
    """
    img = cv2.imread(path)
    img = cv2.resize(img, (h, w))
    if(to_grey):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255    
    return img   
