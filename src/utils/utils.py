""" Utilities
e.g. for data analysis + processing
"""
import os, sys, time, datetime, texttable, re
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
    string = replace_special_chars(string, '_')
    return string[0].upper() + string[1:]


def replace_special_chars(string, char='_'):
    return re.sub('[^A-Za-z0-9]+', char, string)


def concat(ls):
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
    s = replace_special_chars(string, char='').lower()
    porter = nltk.PorterStemmer()
    print(s)
    s_ = [porter.stem(t) for t in nltk.word_tokenize(s)]
    return intersperse(s_)
