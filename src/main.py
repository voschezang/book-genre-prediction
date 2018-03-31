import os
os.chdir('src/')
print(os.getcwd())

import pandas, os, numpy as np, collections
import os, sklearn, pandas, numpy as np
from sklearn import svm
import skimage
from skimage import io, filters
from utils import utils  # custom functions, in local environment
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
## NN libs
import keras
from sklearn.decomposition import PCA
from keras.utils import to_categorical
from keras.layers import Dense, Activation
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Conv3D, MaxPool2D, Dropout, Flatten

import data, config, tfidf, models
from utils import io

# info = pandas.read_csv(config.dataset_dir + 'final_data.csv')
dataset = data.init_dataset()
dataset.info.keys()
