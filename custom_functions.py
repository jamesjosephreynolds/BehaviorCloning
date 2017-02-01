''' This is a utility file to define functions
    model.py will reference these functions to
    have a clean layout and improved readability '''

import numpy as np
import csv
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def check_custom_functions_import():
    print("custom_functions.py loaded successfully")

def get_list_from_csv(file):
    # get a list of elements from a csv file

    with open(file, 'r') as f:
        reader = csv.reader(f)
        return_list = list(reader)

    return return_list

def get_col_data_from_list(list_data, col_idx, row_idx):
    # get column data from list as array of real numbers
    # this applies to the steering angle data in the csv

    col_np_array = np.zeros((1), dtype = np.float32)
    col_np_array = float(list_data[row_idx][col_idx])

    return col_np_array


def get_img_file_from_list(list_data, col_idx, row_idx):
    # get image data from list as array of images
    # this applies to the filenames of images in the csv

    img_list = list_data[row_idx][col_idx]

    return img_list

def get_img_from_file(file):
    # take filename and path and load into a numpy array
    # this will actually load the images data, not filename
    
    img = cv2.imread(file, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img
    
def norm_data(data):
    # normalize image data in uint8 type
    # range is -0.5 to 0.5
    
    norm_data = np.zeros_like(data, dtype = np.float32)
    norm_data = data/255 - 0.5

    return norm_data

def crop_border(img, shiftV, shiftH, size = (66,200)):
    # crop the image inside the remaining non-black area
    # after vertical and horizontal translation
    
    rows, cols, _ = img.shape

    # find the center pixel of the input image
    center = (rows/2, cols/2)

    # find the corners of the smaller image
    # within the valid remaining area
    y0 = np.int(center[0] - size[0]/2)
    y1 = np.int(center[0] + size[0]/2)
    x0 = np.int(center[1] - size[1]/2)
    x1 = np.int(center[1] + size[1]/2)

    # grab the smaller image
    img = img[y0:y1, x0:x1]

    # redundant resize in case rounding generates +/- 1 pixel error
    img = resize(img, size[1], size[0])
    
    return img

def crop_hood_sky(img, hood, sky):
    # crop the hood and the sky out of the image
    rows, cols, _ = img.shape

    y0 = sky
    y1 = rows - hood

    # grab the smaller image
    img = img[y0:y1, 0:cols]
  
    return img

def vertical_shift(img, maxY = 30):
    # return the image, randomly shifted up to max pixels
    # return shift as feedback indicating the magnitude
    
    rows, cols, _ = img.shape
    
    # 31 possible steps to shift, scaled by maxY
    shift = np.int(maxY/15)*np.int(np.random.uniform(-15, 15, 1))
    
    # affine warp matrix M
    M = np.float32([[1, 0, 0],[0, 1, shift]])
    img = cv2.warpAffine(img, M, (cols, rows))

    return img, shift

def horizontal_shift(img, maxX = 40):
    # return the image, randomly shifted up to max pixels
    # return shift as feedback indicating the magnitude
    
    rows, cols, _ = img.shape
    
    # 31 possible steps to shift, scaled by maxX
    shift = np.int(maxX/15)*np.int(np.random.uniform(-15, 15, 1))
    
    # affine warp matrix M
    M = np.float32([[1, 0, shift],[0, 1, 0]])
    img = cv2.warpAffine(img, M, (cols, rows))

    return img, shift

def brightness_shift(img, maxB = 1.25):
    # return the image, randomly scaled in brightness
    
    # maxB < 1 not properly defined for the below shift
    # variable implementation
    maxB = max(maxB, 1)
    shift = np.random.uniform(2-maxB, maxB, 1)
    
    # only shift the third plane (V in HSV space)
    img[:,:,2] = img[:,:,2]*shift

    return img, shift

def resize(img, width = 120, height = 120):
    # resize the image
    
    img = cv2.resize(img,(width, height), interpolation = cv2.INTER_CUBIC)
    
    return img

def get_data_arrays(array):
    # load image names and steering angles into data arrays
    
    N = len(array)
    
    # initialize
    y_array = np.zeros((1, 1), dtype = np.float32)
    
    for idx in range(N):

        x_file = get_img_file_from_list(array, 0, idx)
        y = get_col_data_from_list(array, 3, idx)

        y = np.array([y], dtype = np.float32)
            
        if idx is 0:
            X_array = [x_file]
            y_array[0] = y
        else:
            X_array.append(x_file)
            y_array = np.append(y_array, y)
                
    return X_array, y_array

def pre_process(img, steer, gain, maxY = 20, maxX = 40, maxB = 1.25, hood = 15, sky = 20, size = (66, 200), mode = 'train'):
    # preprocess images
    # in mode == 'train': crop hood and sky, shift vertical, shift horizontal, scale brightness, crop border, flip
    # in mode != 'train': crop hood and sky, crop border

    img = crop_hood_sky(img, hood, sky)
    if mode is 'train':
        img, shiftV = vertical_shift(img, maxY)
        img, shiftH = horizontal_shift(img, maxX)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img, shiftB = brightness_shift(img, maxB)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        if maxX is not 0:
            steer += (shiftH * gain / maxX) # adjust steering angle for horizontal shift
        else:
            steer = steer
    else:
        shiftV = 0
        shiftH = 0
        shiftB = 1
        steer = steer

    img = crop_border(img,  shiftV, shiftH)
    flip_prob = np.random.uniform(-1, 1, 1)
    if flip_prob > 0 and mode is 'train':
        img = cv2.flip(img, 1)
        steer = -steer # flip steering angle while flipping image
    
    return img, steer, shiftV, shiftH, shiftB
