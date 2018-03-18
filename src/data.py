""" Functions that are specific to our dataset
"""
import pandas, collections, os
import sklearn, skimage, skimage.io, pandas, numpy as np
import keras.utils
# from sklearn import svm
# from skimage import data, io, filters
from collections import namedtuple
from utils import utils, io

import config, tfidf

Dataset = namedtuple('Dataset', [
    'info', 'labels', 'genres', 'book_sentiment_words_list', 'label_dataset',
    'sentiment_dataset'
])

SubDataset = namedtuple('SubDataset',
                        ['dict_index_to_label', 'dict_label_to_index'])

# ['train', 'test', 'labels', 'dict_index_to_label', 'dict_label_to_index'])

print(""" Dataset :: namedtuple(
  'info': pandas.df
  'labels': pandas.df('filename.txt': 'genre')
  'genres': ['genre'] # unique genres
  'label_dataset': SubDataset
  'sentiment_dataset': SubDataset
  'book_sentiment_words_list': ['filename']

 SubDataset :: namedtuple(
   'dict_index_to_label' = dict to convert label_index -> label_name
   'dict_label_to_index'= dict to convert label_name -> label_index
""")

#     ['train' = ['img_name']
#     , 'test' = ['img_name']
#     , 'labels' = pandas.df('img_name','breed')
#     , 'dict_index_to_label' = dict to convert label_index -> label_name
#     , 'dict_label_to_index'= dict to convert label_name -> label_index
#     """)


def init_dataset():
    # alt: use utils.Dataset
    # labels = pandas.read_csv(config.dataset_dir + 'labels.csv')
    # train = os.listdir(config.dataset_dir + 'train/')
    # test = os.listdir(config.dataset_dir + 'test/')

    info = pandas.read_csv(config.dataset_dir + 'final_data.csv')
    labels = pandas.read_csv(config.dataset_dir + 'labels.csv')
    genres = io.read_unique_genres()

    label_dataset = init_sub_dataset(genres)

    # lists of files
    book_sentiment_words_list = os.listdir(
        config.dataset_dir + 'output/sentiment_word_texts')

    # feature selection
    # 1. tfidf on sentiment words (most important sentiment words that define genres)
    sentiment_words = ['a', 'b', 'c']  # readfile('sentiment_words.csv')
    sentiment_dataset = init_sub_dataset(sentiment_words)

    # return data as a namedtuple
    return Dataset(info, labels, genres, book_sentiment_words_list,
                   label_dataset, sentiment_dataset)


def init_sub_dataset(word_list):
    # create a label dicts to convert labels to numerical data and vice versa
    # the order is arbitrary, as long as we can convert them back to the original classnames
    # unique_labels = set(labels['breed'])
    dict_index_to_label_ = dict_index_to_label(word_list)
    dict_label_to_index_ = dict_label_to_index(word_list)
    return SubDataset(dict_index_to_label_, dict_label_to_index_)


def extract_genres(info, book_list):
    labels = {}  # {bookname: [genre]}, with max 1 genre
    for filename in book_list[:]:
        # name = filename.split('.')[0]
        book = info.loc[info['filename'] == filename]
        genre = book.genre.item()
        labels[str(filename)] = [genre]
    return labels


def extract_all(dataset, names):
    # Collect test data (+labels)
    dirname = config.dataset_dir + 'output/sentiment_word_texts/'
    x = []
    y = []
    for name in names:
        dataset.labels
        tokenized = io.read_book(dirname, name)
        x.append((name, tokenized))
        y.append(get_label(name, dataset.labels))
    return x, y


def labels_to_vectors(sub_dataset, train_labels, test_labels):
    # dataset contains dicts to convert
    train = textlabels_to_numerical(sub_dataset, train_labels)
    test = textlabels_to_numerical(sub_dataset, test_labels)
    y_train = keras.utils.to_categorical(train)
    y_test = keras.utils.to_categorical(test)
    return y_train, y_test


def tokenlist_to_vector(tokens, sub_dataset):
    # TODO depending on len(tokens)
    selected_words = list(sub_dataset.dict_label_to_index.keys())
    n = len(selected_words)
    if n < 1:
        return None
    counter = collections.Counter(tokens)
    vector = np.zeros([n])
    for i, word in enumerate(selected_words):
        try:
            x = counter[word]
            vector[i] = (x / float(n))**0.5
        except:  # KeyError
            continue
    return vector


def polarization_scores_to_vector(dataset, name='706.txt'):
    row = dataset.info.loc[dataset.info['filename'] == name]
    keys = ['pos score', 'neg score', 'neu score', 'comp score']
    v = []
    for key in keys:
        v.append(row[key].item())
    return np.array(v)


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


def get_label(name='123.txt', labels=[]):
    # labels :: pandas.df :: { id: breed }
    return labels[name][0]
