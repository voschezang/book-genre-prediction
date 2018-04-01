""" Project parameters & config
"""
import numpy as np
seed = 7
# np.random.seed(seed)

tmp_model_dir = '/tmp/ml_model_books'  # see Makefile/logs

dataset_dir = '../datasets/'
src_dir = '../src/'

sentiment_words_dir = dataset_dir + 'sentiment_word_texts/'
info_file = dataset_dir + 'new_final_data.csv'

default_genre_value = 'unknown'

# verbose

debug_ = False
