import cv2
from glob import glob 
import numpy as np
from tqdm import tqdm,trange
import pandas as pd
import os
import deepdish as dd 
import logging

def load_training_data_h5(train_path = '../Datasets/training_data.h5',shuffle = True):
    logging.info("Loading HDF5 Training Data")
    d = dd.io.load(train_path)

    X = d['X']
    y = d['y']

    logging.info("Loaded data with shapes:")
    logging.info(X.shape)
    logging.info(y.shape)

    #Shuffle the data in case there was some sorting done
    if shuffle:
        logging.info("Shuffling the data")
        idx = np.arange(y.shape[0])
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

    return X,y

def load_specific_images(image_ids,train_path = '../Datasets/train/'):
    """Loads the specified images
    
    Args:
        image_ids (np array): array of file/image IDs
        train_path (str, optional): Defaults to '../Datasets/train/'. Path to the training data
    
    Returns:
        np array: loaded images as uint8
    """

    #Allocate numpy array for images
    N = image_ids.shape[0]
    X = np.zeros([N,96,96,3],dtype=np.uint8)

    #Load the images
    for i,image_id in tqdm(enumerate(image_ids), total=N):
        X[i] = cv2.imread(train_path + image_id + ".tif" )

    return X

def load_training_data(train_path = '../Datasets/train/',shuffle = True, N = -1):
    """Reads the training data from the data folder
        train_path (str, optional): Defaults to '../Datasets/train/'. Location of the training images
        shuffle (bool, optional): Defaults to True. Shuffle data after
    
    Returns:
        Numpy arrays: Training images (N,H,W,C) and the corresponding labels (N)
    """

    #Read the provided csv file and image filenames
    df = pd.DataFrame({'path': glob(os.path.join(train_path,'*.tif'))})
    if os.name == 'nt': #deal with windows backslashes
        df['id'] = df.path.map(lambda x: x.split('\\')[1].split(".")[0])
    else:
        df['id'] = df.path.map(lambda x: x.split('/')[3].split(".")[0])
    labels = pd.read_csv("../Datasets/train_labels.csv")
    df = df.merge(labels, on = "id")

    #Allocate numpy arrays for images and labels
    if N == -1:
        N = df["path"].size
    X = np.zeros([N,96,96,3],dtype=np.uint8)
    y = np.squeeze(df.as_matrix(columns=['label']))[0:N]

    #Load the images
    for i, row in tqdm(df.iterrows(), total=N):
        if i == N:
            break
        X[i] = cv2.imread(row['path'])

    #Shuffle the data in case there was some sorting done
    if shuffle:
        idx = np.arange(y.shape[0])
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

    return X,y