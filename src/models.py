import numpy as np
import config  # incl. random seed
np.random.seed(config.seed)
print("NP - - -", np.random.random(2))
# import nn libs
import keras
from keras.utils import to_categorical
from keras.models import Sequential, model_from_json
from keras import optimizers
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model

import data


def load_model(filename, weights, v=False):
    with open(filename, 'r') as json:  # cnn_transfer_augm
        loaded_model_json = json.read()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights)
    if v: print("Loaded model from disk")

    # reset seed ?
    np.random.seed(config.seed)
    optimizer = optimizers.Adam(lr=0.001)
    loaded_model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=['accuracy'])
    if v: print('compiled model')
    return loaded_model


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
    model.add(Dense(512, activation='relu', input_shape=input_shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(output_length, activation='softmax'))
    # in addition, return a function that displays information about the model
    return model, model.summary
