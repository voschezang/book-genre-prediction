from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
from nltk.tokenize import sent_tokenize
import pandas as pd
from wordcloud import WordCloud
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import config
import tfidf


def make_lex_dict(lexicon_file):
    """
    Convert lexicon file to a dictionary
    """
    lex_dict = {}
    for line in lexicon_file.split('\n'):
        (word, measure) = line.strip().split('\t')[0:2]
        lex_dict[word] = float(measure)
    return lex_dict


def return_sentiment_scores(sentence, analyser):
    # return just the sentiment scores
    snt = analyser.polarity_scores(sentence)
    return snt


def per_book(lines):
    analyser = SentimentIntensityAnalyzer()
    sub_pos_list = []
    sub_neg_list = []
    sub_neu_list = []
    sub_comp_list = []
    for line in lines:
        scores = return_sentiment_scores(line, analyser)
        # save sentiment scores
        sub_neg_list.append(scores['neg'])
        sub_neu_list.append(scores['neu'])
        sub_pos_list.append(scores['pos'])
        sub_comp_list.append(scores['compound'])

    # then save average sentiment scores for each book
    neg = sum(sub_neg_list) / float(len(sub_neg_list))
    pos = sum(sub_pos_list) / float(len(sub_pos_list))
    neu = sum(sub_neu_list) / float(len(sub_neu_list))
    comp = sum(sub_comp_list) / float(len(sub_comp_list))
    return (pos, neg, neu, comp)


def sentiment_analysis(directory):
    analyser = SentimentIntensityAnalyzer()
    # returns the sentiment of every book in the directory
    data = pd.read_csv(
        config.dataset_dir + 'output/final_data.csv', index_col=0)
    pos_list = []
    neg_list = []
    neu_list = []
    comp_list = []

    # for every book
    for filename in data['filename']:  # [:max_amt]:

        sub_pos_list = []
        sub_neg_list = []
        sub_neu_list = []
        sub_comp_list = []

        # if file is a textfile
        if filename.endswith(".txt"):
            text = open(
                os.path.join(directory, filename), 'r', errors='replace')
            # for every line in the text
            for line in text.readlines():
                scores = return_sentiment_scores(line, analyser)
                # save sentiment scores
                sub_neg_list.append(scores['neg'])
                sub_neu_list.append(scores['neu'])
                sub_pos_list.append(scores['pos'])
                sub_comp_list.append(scores['compound'])

            # then save average sentiment scores for each book
            neg_list.append((sum(sub_neg_list) / float(len(sub_neg_list))))
            pos_list.append((sum(sub_pos_list) / float(len(sub_pos_list))))
            neu_list.append((sum(sub_neu_list) / float(len(sub_neu_list))))
            comp_list.append((sum(sub_comp_list) / float(len(sub_comp_list))))

    # convert scores to pandas compatible list
    neg = pd.Series(neg_list)
    pos = pd.Series(pos_list)
    neu = pd.Series(neu_list)
    com = pd.Series(comp_list)

    print(len(neg), len(pos), len(neu), len(com))
    # fill the right columns with the right data
    print(type(data), 'type')
    print(neg)
    data['neg score'] = neg.values
    data['pos score'] = pos.values
    data['neu score'] = neu.values
    data['comp score'] = com.values
    data.to_csv(config.dataset_dir + 'output/final_data.csv')
    return data


def count_sentiment_words(directory, sent_dict):
    sent_words_list = []
    pos_list = []
    neg_list = []

    data = pd.read_csv(
        config.dataset_dir + 'output/final_data.csv', index_col=0)

    for filename in data['filename']:
        sent_words_list = []
        pos_count = 0
        neg_count = 0

        if filename.endswith(".txt"):
            text = open(
                os.path.join(directory, filename), 'r', errors='replace')
            sentiment_file = open(
                config.dataset_dir + 'output/sentiment_word_texts/' + filename,
                'w')

            for line in text.readlines():
                for word in line.split(" "):
                    if word in sent_dict:
                        if sent_dict[word] >= 0:
                            pos_count += 1
                            sent_words_list.append(word)
                            sentiment_file.write("%s" % word)
                            sentiment_file.write(" ")
                        else:
                            neg_count += 1
                            sentiment_file.write("%s" % word)
                            sentiment_file.write(" ")

            pos_list.append(pos_count)
            neg_list.append(neg_count)

    data['amt pos'] = pos_list
    data['amt neg'] = neg_list

    data.to_csv(config.dataset_dir + 'output/final_data.csv')
    return data


# count_sentiment_words(config.dataset_dir + 'bookdatabase/books/')


def create_wordcloud(scores, genre):
    font_path = config.dataset_dir + 'Open_Sans_Condensed/OpenSansCondensed-Light.ttf'

    try:
        w = WordCloud(
            background_color='white',
            min_font_size=14,
            font_path=font_path,
            width=1000,
            height=500,
            relative_scaling=1,
            normalize_plurals=False)
        wordcloud = w.generate_from_frequencies(scores)
        wordcloud.recolor(color_func=grey_color_func)

    except ZeroDivisionError:
        return

    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(config.dataset_dir + 'output/wordclouds/' + genre + '.png')
    plt.close()


def grey_color_func(word,
                    font_size,
                    position,
                    orientation,
                    random_state=None,
                    **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(10, 50)


def tfidf_per_genre(plot_wc=False):
    data = pd.read_csv(config.dataset_dir + 'final_data.csv')
    genres_file = open(config.dataset_dir + 'unique_genres.txt', 'r')
    genre_list = [genre.strip('\n') for genre in genres_file.readlines()]
    directory = config.dataset_dir + 'output/sentiment_word_texts/'
    book_list = []

    index = tfidf.create_index(directory)

    for genre in genre_list:
        genre = genre.replace('/', ' ')

        score_dict = {}
        book_list = []

        books_of_genre = data.loc[data['genre'] == genre]

        for book in books_of_genre['filename']:
            book_list.append(book)

        try:
            tf_matrix, genre_tokens = tfidf.create_tf_matrix(
                directory, book_list, genre)

            for term in genre_tokens:
                score = tfidf.tfidf(term, genre, directory, index, tf_matrix)
                score_dict[term] = score

            scores_file = open(config.dataset_dir +
                               'output/top200_per_genre/' + genre + '.txt',
                               'w')

            for w in sorted(score_dict, key=score_dict.get, reverse=True):
                scores_file.write('%s/n' % w)

            scores_file.close()

            print('success')

            if plot_wc:
                font_path = config.dataset_dir + 'Open_Sans_Condensed/OpenSansCondensed-Light.ttf'
                create_wordcloud(score_dict, genre)

        except ZeroDivisionError:
            continue
        except ValueError:
            continue


# return tfidf_dict_per_genre

# tfidf_dict_per_genre = tfidf_per_genre(plot_wc=True)
