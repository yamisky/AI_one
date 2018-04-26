import keras.backend as K
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def load_dataset():
    train_dataset = h5py.File('data_cat_dog.h5', 'r')  
    
    train_set_x_orig = np.array(train_dataset['X_train'][:]) 
    train_set_y_orig = np.array(train_dataset['y_train'][:]) 
    
    test_set_x_orig = np.array(train_dataset['X_test'][:]) 
    test_set_y_orig = np.array(train_dataset['y_test'][:]) 
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig
