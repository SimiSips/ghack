# Import libaries
import tensorflow as tf 
from tensorflow.keras.preprocessing import ImageDataGenerator
import matplotlib.pyplot as plt 
import splitfolders
from tensorflow.keras.callbacks import TensorBoard
import time 

# Create a tensorboard callback
NAME = "DeepHospitalCNN-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

# Use split folders library to split into train and test set
splitfolders.ratio("dataset", "new_data", ratio=(0.9, 0.1))