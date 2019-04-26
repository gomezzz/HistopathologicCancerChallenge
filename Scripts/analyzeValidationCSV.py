import sys
sys.path.append('../') 
import pandas as pd
import cv2,os
from glob import glob 
import numpy as np
from tqdm import tqdm,trange
from Analysis.analyze_validation_results import *
from IO.load_training_data import *

train_path = '../Datasets/train/'

#load csv with predictions and ground truth
csv = pd.read_csv("../Datasets/Resnet18 validation data.csv")

#how many images should we look at?
N = csv['y_pred'].size
# N = 25000

print(csv.head(5))

predictions = csv['y_pred'].values[:N]
labels = csv['y_true'].values[:N]

#load the images
X = load_specific_images(csv['id'].values[:N])

analyze_validation_results(X,labels,predictions)


