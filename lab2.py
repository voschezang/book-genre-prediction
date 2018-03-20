from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import random
import TKinter as tk
def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(10, 50)

f = open('/Users/Tristan/Downloads/const.txt')
text = f.read()
font_path = '/Users/Tristan/books/datasets/Open_Sans_Condensed/' + 'Open_Sans_Condensed/OpenSansCondensed-Light.ttf'

from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))
# w = WordCloud()
w = WordCloud(stopwords=stopWords, background_color='white', min_font_size=14, max_words=1000 ,font_path=font_path ,normalize_plurals=True)
wordcloud = w.generate(text)
wordcloud.recolor(color_func=grey_color_func)
# plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('/Users/Tristan/books/datasets/output/wordclouds/cloud2.png')
plt.close()
