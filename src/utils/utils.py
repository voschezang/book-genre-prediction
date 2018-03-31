""" Utilities
e.g. for data analysis + processing
"""
import os, sys, time, datetime, texttable, re, nltk
import numpy as np, statistics, collections, pandas
from scipy import stats

import config

### ------------------------------------------------------
### Functions
### ------------------------------------------------------


def stem(filename='aed285c5eae61e3e7ddb5f78e6a7a977.jpg'):
    # rmv file extension (.jpg)
    return filename.split('.')[0]


def sanitize(string):
    string = re.sub('"', '', string)
    string = re.sub('`', '', string)
    string = re.sub("'", '', string)
    string = replace_special_chars(string, '_')
    return string[0].upper() + string[1:]


def to_upper(string):
    return string[0].upper() + string[1:]


def replace_special_chars(string, char='_'):
    # keep spaces
    return re.sub('[^A-Za-z0-9]+', char, string)


def replace_special_chars2(string, char='_'):
    return re.sub('[^A-Z a-z0-9]+', char, string)


def concat(ls):
    if ls == []:
        return []
    result = ls[0]
    for char in ls[1:]:
        result += char
    return result


def intersperse(ls, a=' '):
    result = ls[0]
    for char in ls[1:]:
        result += a + char
    return result


def normalize_string(string):
    s = replace_special_chars2(string, char='').lower()
    porter = nltk.PorterStemmer()
    s_ = [porter.stem(t) for t in nltk.word_tokenize(s)]
    return intersperse(s_, ' ')


def rmv_words(text='', words=[]):
    # remove specific words in a string
    tokens = text.split(' ')
    if len(tokens) < 2:
        return concat(tokens)
    # check for max 2 words
    if tokens[-1] in words:
        tokens = tokens[:-1]
        return rmv_words(concat(tokens))
    return intersperse(tokens, ' ')


def find_first(tokens, stem_list):
    for token in tokens:
        if token in stem_list:
            return [token]
    return tokens


def stem_conditionally(text='', stem_list=['']):
    tokens = text.split(' ')
    if len(tokens) > 1:
        tokens = find_first(tokens, stem_list)
        for token in tokens:
            if tokens in stem_list:
                tokens = [tokens[0]]
    return intersperse(tokens, ' ')


def format_score(score):
    return round(score * 100, 2)


def gen_table(th=[], trs=[]):
    # :th :: (k,v)
    # :rows :: [(k,v)]
    ta = texttable.Texttable()
    rows = [th]
    for tr in trs:
        rows.append(tr)
    ta.add_rows(rows)
    return ta.draw()
