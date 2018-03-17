""" Functions that are specific to our dataset
"""
import pandas
import os, sklearn, skimage, skimage.io, pandas, numpy as np
import keras.utils
# from sklearn import svm
# from skimage import data, io, filters
from collections import namedtuple

import config

Dataset = namedtuple(
    'Dataset',
    ['train', 'test', 'labels', 'dict_index_to_label', 'dict_label_to_index'])

print(""" Dataset :: namedtuple(
    ['train' = ['img_name']
    , 'test' = ['img_name']
    , 'labels' = pandas.df('img_name','breed')
    , 'dict_index_to_label' = dict to convert label_index -> label_name
    , 'dict_label_to_index'= dict to convert label_name -> label_index
    """)


def init_dataset():
    # alt: use utils.Dataset
    labels = pandas.read_csv(config.dataset_dir + 'labels.csv')
    train = os.listdir(config.dataset_dir + 'train/')
    test = os.listdir(config.dataset_dir + 'test/')

    # create a label dicts to convert labels to numerical data and vice versa
    # the order is arbitrary, as long as we can convert them back to the original classnames
    unique_labels = set(labels['breed'])
    dict_index_to_label_ = dict_index_to_label(unique_labels)
    dict_label_to_index_ = dict_label_to_index(unique_labels)
    # return data as a namedtuple
    return Dataset(train, test, labels, dict_index_to_label_,
                   dict_label_to_index_)


def labels_to_vectors(dataset, train_labels, test_labels):
    # dataset contains dicts to convert
    train = textlabels_to_numerical(dataset, train_labels)
    test = textlabels_to_numerical(dataset, test_labels)
    y_train = keras.utils.to_categorical(train)
    y_test = keras.utils.to_categorical(test)
    return y_train, y_test


def textlabels_to_numerical(dataset, labels):
    # transform ['label'] => [index]
    # (list of text => list of indices)
    return [dataset.dict_label_to_index[label] for label in labels]


def dict_index_to_label(labels):
    # labels :: list or set()
    # return { int: label }
    unique_labels3 = set(labels)
    return {k: v for k, v in enumerate(unique_labels3)}


def dict_label_to_index(labels):
    # labels :: list or set()
    # return { label: int }
    unique_labels = set(labels)
    return {k: v for v, k in enumerate(unique_labels)}


def get_label(img_name='aed285c5eae61e3e7ddb5f78e6a7a977.jpg', labels=[]):
    # labels :: pandas.df :: { id: breed }
    # index_dict :: { value: index } :: { breed: int }
    label = labels.loc[labels['id'] == utils.stem(img_name)]
    return label.breed.item()
