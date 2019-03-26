from os import walk
from PIL import Image
import numpy as np
import json

def get_data(data_directory):
    '''Input data directory. Returns train directory, validation directory and test directory'''
    train_dir = data_directory + '/train'
    valid_dir = data_directory + '/valid'
    test_dir = data_directory + '/test'

    return train_dir, valid_dir, test_dir

def label_count(path):
    '''Input filepath. Returns count of subdirectories (or distinct labels)'''
    count = 0
    for root, dirs, files in walk(path):
            count += len(dirs)
    
    return count

def get_labels(labels_filepath):
    '''Input labels filepath (JSON object).  Returns Dictionary of Integer Encoded Labels (keys) and actual names (values).  Returns None if no labels filepath provided'''
    if labels_filepath == None:
        labels = None
    else:
        with open(labels_filepath, 'r') as f:
            labels = json.load(f) 
    
    return labels

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Numpy array'''
    
    x = Image.open(image)
    
    aspect_ratio = x.width / x.height
    if x.width < x.height:  # if width is smallest size, set to 256 and maintain aspect ratio
        y = x.resize((256, int(256 / aspect_ratio)))
    else:   # if height is smallest size, set to 256 and maintain aspect ratio
        y = x.resize((int(256 * aspect_ratio), 256))

    crop_side = 224 # square crop side length
    crop_upper_left = (0 + (y.width - crop_side) / 2, # upper left coordinate for crop
                       0 + (y.height - crop_side) / 2)
    crop_lower_right = (y.width - (y.width - crop_side) / 2, # lower right coordinate for crop
                        y.height - (y.height - crop_side) / 2)
    cropped_y = y.crop(crop_upper_left + crop_lower_right) # center-cropped image
    
    np_image = np.array(cropped_y) # image as numpy array
    np_image = np_image / 255 # values as floats from 0-1
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    np_image = (np_image - mean)/std
        
    return np_image.transpose((2,0,1)) # Color channel dimension as first dimension
