#Append other folders to path
import sys
sys.path.append('../') 

import deepdish as dd
import logging 
import numpy as np

from IO.load_training_data import load_training_data
from Data.augment import augment

def createAugmentedDataset(targetPath = "../Datasets/training_data.h5",augmentationType = "default"):
    """Creates one round of augmentations of each image in the training (can be improved), 
	supports different augmentations, stores to hdf5
        targetPath (str, optional): Defaults to "/Datasets/training_data.h5". [description]
        augmentationType (str, optional): Defaults to "default". [description]
    """

    logging.info("Loading training data")
    X,y = load_training_data() # Loads the training data from /Data/

    logging.info("Augmenting images")
    X_aug = augment(X) # Augment each image once

    logging.info("Appending augmented images")
    X = np.concatenate((X,X_aug)) # Add the augmented images to the data
    y = np.concatenate((y,y)) # Add the labels for the augmented images

    logging.info("Storing HDF5 dataset")
    data = {'X': X, 'y': y}
    dd.io.save(targetPath, data, compression=('blosc', 8))

    logging.info("Created augmented HDF5 dataset")