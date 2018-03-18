""" Functions that are specific to our dataset
"""
import pandas
import os, sklearn, skimage, skimage.io, pandas, numpy as np
import keras.utils
# from sklearn import svm
# from skimage import data, io, filters
from collections import namedtuple

import config

Dataset = namedtuple('Dataset',
                     ['info', 'labels', 'book_sentiment_words_list'])

# ['train', 'test', 'labels', 'dict_index_to_label', 'dict_label_to_index'])

print(""" Dataset :: namedtuple(
  'info': pandas.df
  'labels': pandas.df('filename.txt': 'genre')
  'book_sentiment_words_list': ['filename']
""")

#     ['train' = ['img_name']
#     , 'test' = ['img_name']
#     , 'labels' = pandas.df('img_name','breed')
#     , 'dict_index_to_label' = dict to convert label_index -> label_name
#     , 'dict_label_to_index'= dict to convert label_name -> label_index
#     """)


def read_unique_genres():
    genres_file = open(config.dataset_dir + 'unique_genres.txt', 'r')
    return [genre.strip('\n') for genre in genres_file.readlines()]


def init_dataset():
    # alt: use utils.Dataset
    # labels = pandas.read_csv(config.dataset_dir + 'labels.csv')
    # train = os.listdir(config.dataset_dir + 'train/')
    # test = os.listdir(config.dataset_dir + 'test/')

    info = pandas.read_csv(config.dataset_dir + 'final_data.csv')
    labels = pandas.read_csv(config.dataset_dir + 'labels.csv')

    # lists of files
    book_sentiment_words_list = os.listdir(
        config.dataset_dir + 'output/sentiment_word_texts')

    # create a label dicts to convert labels to numerical data and vice versa
    # the order is arbitrary, as long as we can convert them back to the original classnames
    # unique_labels = set(labels['breed'])
    # dict_index_to_label_ = dict_index_to_label(unique_labels)
    # dict_label_to_index_ = dict_label_to_index(unique_labels)
    # return data as a namedtuple
    # return Dataset(train, test, labels, dict_index_to_label_,
    #                dict_label_to_index_)
    return Dataset(info, labels, book_sentiment_words_list)


def extract_genres(info, book_list):
    labels = {}  # {bookname: [genre]}, with max 1 genre
    for filename in book_list[:]:
        # name = filename.split('.')[0]
        book = info.loc[info['filename'] == filename]
        genre = book.genre.item()
        labels[str(filename)] = [genre]
    return labels


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
