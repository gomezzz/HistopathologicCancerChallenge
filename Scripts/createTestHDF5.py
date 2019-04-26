#Append other folders to path
import sys
sys.path.append('../') 

from IO.load_test_data import load_test_data
import logging
import deepdish as dd

logging.basicConfig(level=0)

X = load_test_data()
targetPath = "../Datasets/test_data.h5"

logging.info("Storing HDF5 dataset")
data = {'X': X}
dd.io.save(targetPath, data, compression=('blosc', 8))
