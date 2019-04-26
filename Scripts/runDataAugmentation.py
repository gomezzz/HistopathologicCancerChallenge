import sys
sys.path.append('../') 

import logging

from Data.createAugmentedDataset import *

logging.basicConfig(level=0)
createAugmentedDataset()