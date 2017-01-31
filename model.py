'''
This script takes CSV data from the driving simulator
and uses the Keras module to train a CNN that outputs
desired steering angle
'''

'''Import modules'''
import numpy as np
import tensorflow as tf
import cv2
import custom_functions as cf
from keras.layers.core import Flatten, Dense, Activation, Dropout
from keras.activations import relu, softmax
from keras.models import Model, Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import json
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

cf.check_custom_functions_import()

'''Load data using utility module custom_functions.py'''
drive_data = cf.get_list_from_csv('driving_log.csv')
X_data_file, y_data = cf.get_data_arrays(drive_data)
X_train_file, X_validation_file, y_train, y_validation = train_test_split(X_data_file, y_data,
                                        test_size = 0.2, random_state=0)


if 1: #debug switch visualize pre-processing

    X_data_in = cf.get_img_from_file(X_train_file[0])
    y_data_in = y_train[0]
    fig = plt.figure()
    ax1 = fig.add_subplot(4,4,1)
    ax1.imshow(X_data_in)
    ax1.set_xlabel('steer = '+str(y_data_in))
    
    for loop in range(2,17):
        X_data, y_data, shiftV, shiftH, shiftB, = cf.pre_process(X_data_in, y_data_in, 0.5)
        axN = fig.add_subplot(4,4,loop)
        axN.imshow(X_data)
        axN.set_xlabel('steer = '+str(y_data))
        
    plt.savefig('augmentation.png',format = 'PNG')

'''Define data generator'''
def data_generator(X_file, y, batch_size = 256, width = 200, height = 66):

    X_batch = np.zeros((batch_size, height, width, 3), dtype = np.float32)
    y_batch = np.zeros(batch_size, dtype = np.float32)

    N = len(y)
    data_idx = 0
    while 1:
        for batch_idx in range(batch_size):
            if data_idx >= N:
                data_idx = 0

            keep_prob = 0
            while keep_prob < 0.75:
                steer = y[data_idx]
                img = cf.get_img_from_file(X_file[data_idx])
                if steer < 0.1:
                    keep_prob = np.random.uniform(0,1,1)
                    data_idx += 1
                    if data_idx >= N:
                        data_idx = 0
                else:
                    keep_prob = 1
                
            if steer < 0.1:
                maxX = 30
            elif steer < 0.2:
                maxX = 20
            elif steer < 0.3:
                maxX = 10
            else:                      
                maxX = 0

            img, steer, _, _, _ = cf.pre_process(img, steer, 0.7, maxX = maxX)

            X_batch[batch_idx] = cf.norm_data(img)
            y_batch[batch_idx] = steer
            data_idx += 1

        yield X_batch, y_batch

'''Define CNN model architecture with Keras
    use example from Nvidia End to End Learning for Self-Driving Cars
    as reference for the initial model architecture
    use max pooling instead of 2x2 stride, as suggested by Vivek
    use 3x3 kernel instead of 5x5 for computational simplicity
    start with 3 convolution layers, 3 fully connected'''
drive_model = Sequential()

drive_model.add(Convolution2D(24, 5, 5, input_shape = (66, 200, 3), activation='relu'))
drive_model.add(MaxPooling2D((2, 2), border_mode = 'same'))
drive_model.add(Dropout(0.5))

drive_model.add(Convolution2D(36, 5, 5, activation='relu'))
drive_model.add(MaxPooling2D((2, 2), border_mode = 'same'))
drive_model.add(Dropout(0.5))

drive_model.add(Convolution2D(48, 5, 5, activation='relu'))
drive_model.add(MaxPooling2D((2, 2), border_mode = 'same'))
drive_model.add(Dropout(0.5))

drive_model.add(Convolution2D(64, 3, 3, activation='relu'))
drive_model.add(Dropout(0.5))

drive_model.add(Convolution2D(64, 3, 3, activation='relu'))
drive_model.add(Dropout(0.5))

drive_model.add(Flatten())

drive_model.add(Dense(100,  activation='relu'))
drive_model.add(Dropout(0.5))

drive_model.add(Dense(50,  activation='relu'))
drive_model.add(Dropout(0.5))

drive_model.add(Dense(10, activation='tanh'))
drive_model.add(Dropout(0.5))

drive_model.add(Dense(1))

''' Compile and reload the model'''
drive_model.compile(optimizer='Adam', loss='mse', lr = 0.00002)
drive_model.load_weights('model.h5', by_name=False)

if 1: # debug switch train model
    '''Train the model'''
    drive_model.fit_generator(data_generator(X_train_file, y_train),
                              validation_data = data_generator(X_validation_file, y_validation),
                              nb_val_samples = 2048, samples_per_epoch = 32768,
                              nb_epoch = 3, verbose = 1)

if 1: # debug switch show model summary
    drive_model.summary()
    my_weights = drive_model.get_weights()
    print(my_weights[0][0])

drive_model.save_weights('model.h5')
model_json = drive_model.to_json()
json.dump(model_json, open('model.json', 'w+'))
