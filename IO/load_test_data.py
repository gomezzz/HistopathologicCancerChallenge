import cv2
from glob import glob 
import numpy as np
from tqdm import tqdm,trange
import pandas as pd
import os
import deepdish as dd 
import logging

def load_test_data_h5(test_path = '../Datasets/test_data.h5'):
    """Reads the test data from an HDF5 file. Use /Scripts/createTestHDF5.py to create the HDF5
        test_path (str, optional): Defaults to '../Datasets/test_data.h5'. path to the hdf5
    
    Returns:
        np array: loaded images
    """

    logging.info("Loading HDF5 Test Data")
    d = dd.io.load(test_path)

    X = d['X']

    logging.info("Loaded data with shapes:")
    logging.info(X.shape)

    return X


def load_test_data(test_path = '../Datasets/test/'):
    """Reads the test data from the data folder
        test_path (str, optional): Defaults to '../Datasets/test/'. Location of the test images
    
    Returns:
        Numpy arrays: Test images (N,H,W,C)
    """

    #Read the provided csv file and image filenames
    df = pd.DataFrame({'path': glob(os.path.join(test_path,'*.tif'))})
    if os.name == 'nt': #deal with windows backslashes
        df['id'] = df.path.map(lambda x: x.split('\\')[1].split(".")[0])
    else:
        df['id'] = df.path.map(lambda x: x.split('/')[3].split(".")[0])

    #Allocate numpy arrays for images and labels
    N = df["path"].size
    X = np.zeros([N,96,96,3],dtype=np.uint8)

    #Load the images
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        X[i] = cv2.imread(row['path'])

    return X