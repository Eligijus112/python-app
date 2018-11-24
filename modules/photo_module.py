"""
The methods in this script will deal with images
"""

import cv2
import os
import re

def img_read(path, append_path = True):
    all_photo = os.listdir(path)
    good_img = []
    for f in all_photo:
        if append_path: 
            f = path + '/' + f
        if re.search(r'.jpg$|.png$|.jpeg$', f) is not None:
            good_img.append(f)
    return good_img
    
def img_preproc(path, h, w, to_grey = True):
    img = cv2.imread(path)
    img = cv2.resize(img, (h, w))
    if(to_grey):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255    
    return img   
