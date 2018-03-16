import numpy as np
from string import punctuation
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.snowball import EnglishStemmer
import math, os
from collections import defaultdict, Counter
from math import log10, sqrt


def tokenize(text):
    result = []
    tokenized = word_tokenize(text)
    for token in tokenized:
        result.append(token.lower())

    return result


def create_index(directory, book_list):
    index = defaultdict(set)

    for filename in book_list:
        if filename.endswith(".txt"):
            text = open(os.path.join(directory, filename), 'r', errors='replace').read()
            for term in tokenize(text):
                index[term].add(filename)

    return index


def create_tf_matrix(directory, book_list):
    tf_matrix = defaultdict(Counter)

    for filename in book_list:
        if filename.endswith(".txt"):
            text = open(os.path.join(directory, filename), 'r', errors='replace').read()
            tokens = tokenize(text)
            tf_matrix[filename] = Counter(tokens)

    return tf_matrix


def tf(t,d, tf_matrix):
    return float(tf_matrix[d][t])


def df(t, index):
    return float(len(index[t]))


def idf(t, book_list, index):
    num_documents = float(len(book_list))
    return log10(num_documents / float(df(t, index)))


def tfidf(t, d, book_list, index, tf_matrix):
    return tf(t, d, tf_matrix) * idf(t, book_list, index)


def perform_tfidf(directory, book_list, index, tf_matrix):
    tfidf_dict = defaultdict
    for filename in book_list:
        if filename.endswith(".txt"):
            text = open(os.path.join(directory, filename), 'r', errors='replace').read()
            for term in tokenize(text):
                    tfidf_score = (tfidf(term, filename, book_list, index, tf_matrix))
                    tfidf_dict[term] = tfidf_score

    return tfidf_dict