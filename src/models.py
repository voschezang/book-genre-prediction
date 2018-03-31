# import nn libs
import keras, numpy as np
from sklearn.decomposition import PCA
from keras.utils import to_categorical
from keras.models import Sequential, model_from_json
from keras import optimizers
from keras.layers import Dense, Activation
from keras.optimizers import SGD, Adam

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten

from keras.models import Model

import data


def load_model(filename, weights):
    with open(filename, 'r') as json:  # cnn_transfer_augm
        loaded_model_json = json.read()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights)
    print("Loaded model from disk")
    optimizer = optimizers.Adam(lr=0.001)
    loaded_model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=[
            'accuracy', 'mean_squared_error', 'categorical_crossentropy',
            'top_k_categorical_accuracy'
        ])
    print('compiled model')
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
