# import nn libs
import keras, numpy as np
from sklearn.decomposition import PCA
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, Adam

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten

from keras.models import Model

import data


def to_vector(x_train, dataset):
    x2 = []
    for filename, tokens in x_train:
        v1 = data.tokenlist_to_vector(tokens, dataset.sentiment_dataset)
        v2 = data.polarization_scores_to_vector(dataset, filename)
        v = np.append(v1, v2)
        x2.append(v)
    return np.stack(x2)


def sequential(input_shape, output_length, dropout=0.10):
    model = Sequential()
    model.add(Dense(255, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(Activation('relu'))  # because relu is awesome
    model.add(Dense(output_length))
    model.add(Activation('softmax'))
    # in addition, return a function that displays information about the model
    return model, model.summary
