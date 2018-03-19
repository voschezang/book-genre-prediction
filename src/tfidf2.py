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


def create_index(genre_list, doc_list):
    index = defaultdict(set)

    for genre, text in zip(genre_list, doc_list):
        genre = genre.replace('/', ' ')
        for term in text:
            index[term].add(genre)
    return index


def create_tf_matrix(genre_list, doc_list):
    # tf_matrix :: {'book': {'word': count} }
    tf_matrix = defaultdict(Counter)

    for genre, text in zip(genre_list, doc_list):
        genre = genre.replace('/', ' ')
        tf_matrix[genre] = Counter(text)

    return tf_matrix


def genre_document(booklist, directory):
    # total_tokens = ''
    total_tokens = []
    for filename in booklist:
        if filename.endswith(".txt"):
            text = open(
                os.path.join(directory, filename), 'r',
                errors='replace').read()
            tokens = tokenize(text)
            for token in tokens:
                # total_tokens += ' ' + token
                total_tokens.append(token)
    genre_tokens = total_tokens

    return genre_tokens


def tf(t, genre, tf_matrix):
    return 1 + log10(float(tf_matrix[genre][t]))


def df(t, index):
    return float(len(index[t]))


def idf(t, doc_list, index):
    num_documents = float(len(doc_list))
    return log10(1 + (num_documents / float(df(t, index))))


def tfidf(t, genre, doc_list, index, tf_matrix):
    score = (tf(t, genre, tf_matrix)) * (idf(t, doc_list, index))
    return score


def perform_tfidf(directory, book_list, index, tf_matrix, genre):
    tfidf_dict = {}
    total_tokens = []

    for filename in book_list:
        if filename.endswith(".txt"):
            # text = open(os.path.join(directory, filename), 'r', errors='replace').read()
            text = open(
                os.path.join(directory, filename), 'r',
                errors='replace').read()

            tokenized = tokenize(text)
            for token in tokenized:
                total_tokens.append(token)

    for term in set(total_tokens):
        score = tfidf(term, genre, directory, index, tf_matrix)
        tfidf_dict[term] = score

    return tfidf_dict


# def

# data = set(tokens van boeken uit genre x)
# for term in data:
#     tf = frequency of term
#     df = log-dinges van term in alle documenten die niet bij genre x horen
#     tfidf = tf / df
