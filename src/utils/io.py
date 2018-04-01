""" Read/write from/to disk
"""
import pandas, os, config

import tfidf, config


def read_book(dirname, filename):
    if not dirname[-1] == '/':
        dirname += '/'
    # text = open(dirname + filename, 'r', errors='replace').read()
    # tokenized = tfidf.tokenize(text)
    return read_book2(dirname + filename)


def read_book2(filename):
    with open(filename, 'r', errors='replace') as book:
        text = book.read()
    # text = open(filename, 'r', errors='replace').read()
    tokenized = tfidf.tokenize(text)
    return tokenized


def read_book3(filename):
    with open(filename, 'r', errors='replace') as book:
        text = book.read()
    tokenized = tfidf.tokenize(text)
    lines = []
    with open(filename, 'r', errors='replace') as book:
        for line in book.readlines():
            lines.append(line)
    return tokenized, lines


def read_unique_genres():
    genres_file = open(config.dataset_dir + 'unique_genres.txt', 'r')
    return [genre.strip('\n') for genre in genres_file.readlines()]


def read_sw_per_genre(amt=1000, dirname='top200_per_genre/'):
    dirname = config.dataset_dir + dirname
    result = []

    print('\n\n\n\n\n\n\n sw dataset')

    for file_ in os.listdir(dirname):
        if not file_ == '_DS_Store':
            with open(dirname + file_, 'r', errors='replace') as ls:
                text = ls.read()
                # text = open(dirname + file_, 'r', errors='replace').read()
                words = text.split('/n')
            # save n words per file
            for w in words[:amt]:
                result.append(w)
    print(result[:20])
    print('<<><<<<')
    for x in range(3):
        a = set(result[:10])
        print(list(a)[:20])
    return set(result)


def print_dict(dirname="", d={}, name="text"):
    if not dirname == "":
        dirname += "/"
    name += ".txt"
    with open(dirname + "0_" + name, "w") as text_file:
        print(name + "\n", file=text_file)
        for k, v in d.items():
            # print(f"{k}:{v}", file=text_file) # pythonw, python3
            print('{:s}, {:s}'.format(str(k), str(v)), file=text_file)


def save_dict_to_csv(dirname, name, data):
    # panda df requires data to be NOT of type {key: scalar}
    if not dirname[-1] == '/':
        dirname += '/'
    filename = dirname + name + ".csv"
    df = pandas.DataFrame(data=data)
    df.to_csv(filename, sep=',', index=False)
    # mkdir filename
    # for k in d.keys(): gen png
    return filename
