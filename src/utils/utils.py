""" Utilities
e.g. for data analysis + processing
"""
import os, sys, time, datetime, texttable, copy, random
import numpy as np, statistics, collections, pandas
from scipy import stats

import config

### ------------------------------------------------------
### Functions
### ------------------------------------------------------


def stem(filename='aed285c5eae61e3e7ddb5f78e6a7a977.jpg'):
    # rmv file extension (.jpg)
    return filename.split('.')[0]
