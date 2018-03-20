from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import random

f = open('/Users/Tristan/Downloads/const.txt')
text = f.read()

w = WordCloud()
wordcloud = w.generate(text)
# plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('/Users/Tristan/books/datasets/output/wordclouds/cloud1.png')
plt.close()
