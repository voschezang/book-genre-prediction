""" Functions that are specific to our dataset
"""
import pandas, collections, os
import sklearn, skimage, skimage.io, pandas, numpy as np
import keras.utils
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


def init_dataset():
    # alt: use utils.Dataset
    # labels = pandas.read_csv(config.dataset_dir + 'labels.csv')
    # train = os.listdir(config.dataset_dir + 'train/')
    # test = os.listdir(config.dataset_dir + 'test/')

    info = pandas.read_csv(config.info_file)
    labels = pandas.read_csv(config.dataset_dir + 'labels.csv')
    genre_list = list(info['genre'])
    genre_list.append(config.default_genre_value)
    genres = set(genre_list)

    label_dataset = init_sub_dataset(genres)

    # lists of files
    book_sentiment_words_list = os.listdir(config.sentiment_words_dir)

    # feature selection
    # 1. tfidf on sentiment words (most important sentiment words that define genres)
    sentiment_words = io.read_sw_per_genre(
        amt=1000, dirname='top200_per_genre/')
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
        if book.empty:
            labels[str(filename)] = config.default_genre_value
        else:
            genre = book.genre.item()
            labels[str(filename)] = [genre]
    return labels


def extract_all(dataset, names):
    # Collect test data (+labels)
    # TODO use actual dir
    dirname = config.sentiment_words_dir
    x = []
    y = []
    for name in names:
        tokenized = io.read_book(dirname, name)
        x.append((name, tokenized))
        y.append(get_label(name, dataset.labels))
    return x, y


def labels_to_vectors(sub_dataset, train_labels, test_labels):
    # dataset contains dicts to convert
    # TODO make sure that every label is present in both y_test and y_test
    train = textlabels_to_numerical(sub_dataset, train_labels)
    test = textlabels_to_numerical(sub_dataset, test_labels)
    y_train = keras.utils.to_categorical(train)
    y_test = keras.utils.to_categorical(test)
    return y_train, y_test


def decode_y(dataset, vector=[], n_best=1):
    dict_ = y_to_label_dict(dataset, vector)
    # best value
    if n_best == 1:
        i = vector.argmax()
        label = dataset.label_dataset.dict_index_to_label[i]
        return dict_, [label]
    else:
        # return n best label predicitions
        ls = list(dict_.items())
        ls.sort(key=lambda x: x[1])
        selected = ls[-1 * n_best:-1]
        return dict_, [label for label, s_ in selected]


def y_to_label_dict(dataset, vector=[]):
    n = vector.shape[0]
    result = {}  # :: {label: score}
    for i in range(n):
        label = dataset.label_dataset.dict_index_to_label[i]
        result[label] = vector[i]
    return result


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
        if key in row and not row.empty:
            v.append(row[key].item())
        else:
            if config.debug_: print('pol key not found', row)
            v.append(0)
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


def normalize_genre(g='horror'):
    # remove keywords such as 'fiction'
    unused_words = ['fiction', 'novel', 'literature', 'literatur']
    known_genres = [
        'children', 'christian', 'fantasi', 'histor', 'horror', 'philosophi',
        'polit', 'western', 'thriller', 'scienc', 'detective', 'apocalypt'
        'romanc'
    ]
    # synonyms and typos (first member is correct)
    synonyms = [('satire', 'satirical'), ('histor', 'histori'), ('young adult',
                                                                 'youngadult'),
                ('fairy tale', 'fairytale'), ('science fiction', 'scienc',
                                              'science'), ('apocalypt',
                                                           'postapocalypt'),
                ('philosophi', 'philosoph'), ('romance', 'romanc', 'romant')]

    # do not confuse 'science' and 'science fiction'
    g = g.lower()
    if g == 'science':
        return g

    g = utils.normalize_string(g)
    g = utils.rmv_words(g, unused_words)
    # remove sub-genres such as 'horror fiction'
    g = utils.stem_conditionally(g, known_genres)
    # remove unclassfiable words such as 'fiction
    g = utils.rmv_words(g, unused_words)
    # remove synonyms and typos
    for tuple_ in synonyms:
        if g in tuple_:
            g = tuple_[0]
    return g


def reduce_genres(genres=['']):
    if type(genres) is str:
        return normalize_genre(genres)
    # g_ = set([utils.normalize_string(g) for g in genres])
    return set([normalize_genre(g) for g in genres])


###### Data analysis

# def analyse_ml_result(dataset, y_test, results, n_best=1):
#     correct = 0
#     incorrect = 0
#     for i, label in enumerate(y_test):
#         all_, best = decode_y(dataset, results[i], n_best=n_best)
#         _, label = decode_y(dataset, label, n_best=1)
#         if label[0] in best:
#             correct += 1
#         else:
#             incorrect += 1
#     return correct, incorrect


def analyse_ml_result(dataset, y_test, results, n_best=1):
    correct = 0
    incorrect = 0
    correct_labels = []
    incorrect_labels = []
    for i, label in enumerate(y_test):
        _, best = decode_y(dataset, results[i], n_best=n_best)
        _, labels = decode_y(dataset, label, n_best=1)
        label = labels[0]
        if label in best:
            correct += 1
            correct_labels.append(label)
        else:
            incorrect += 1
            incorrect_labels.append(label)
    return correct, incorrect, correct_labels, incorrect_labels
