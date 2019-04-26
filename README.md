# HistopathologicCancerChallenge
This repository contains notebooks and code written by me for the Kaggle Histopathologic Cancer Detection Challenge. My team (Health Hackers) ensembled submissions using the model from this repo and those created by other team members to achieve a Top 3% finish.

# Prerequisites
* imgaug, deepdish, cv2, pandas, tqdm 
* keras

Please place the challenge data in the datasets folder

# Features
* Streaming data augmentation using imgaug
* Utilizes Keras to train a NasNetMobile for the task
* Low VRAM requirements

# Structure
* Analysis - Contains code to analyze results on the validation data
* Data - Contains processing code for the data augmentation
* IO - Code for loading/storing HDF5 datasets
* Models - Contains the implemented models (using NASNetMobile and a simple CNN)
* notebooks - Contains two notebooks to train the model
  * *Keras Starter* - Minimal starter that trains a NASNetMobile using hdf5 data and creates a submission
  * *Keras NASNet* - Trains a NASNetMobile with streaming data augmentation, CLR and TTA
* Out - output will be stored here
* Scripts - Contains scripts to augmented and store data
* Utils - Plotting utilities and a cyclic learning rate for keras
