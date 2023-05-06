from __future__ import print_function, division
from future.utils import iteritems


import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup


wordnet_lemmmatizer = WordNetLemmatizer()
stop_words = set(w.rstrip() for w in open('data/stopwords.txt'))

positive_reviews = BeautifulSoup(open(r"data/electronics/positive.review").read(), features="lxml")
positive_reviews = positive_reviews.find_all('review_text')

negative_reviews = BeautifulSoup(open(r"data/electronics/negative.review").read(), features="lxml")
negative_reviews = negative_reviews.find_all('review_text')

np.random.shuffle(positive_reviews)
positive_reviews = positive_reviews[: len(negative_reviews)]


def my_tokenizer(s):
    s = s.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [wordnet_lemmmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in stop_words]
    return tokens


word_index_map = {}
current_index = 0

positive_tokenized = []
negative_tokenized = []


for review in positive_reviews:
    tokens = my_tokenizer(review.text)
    positive_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1


for review in negative_reviews:
    tokens = my_tokenizer(review.text)
    negative_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1


def tokens_to_vector(tokens, label):
    x = np.zeros(len(word_index_map) + 1)
    for t in tokens:
        i = word_index_map[t]
        x[i] += 1
    x = x / x.sum()
    x[-1] = label
    return x


N = len(positive_tokenized) + len(negative_tokenized)
data = np.zeros((N, len(word_index_map) + 1))
i = 0

for token in positive_tokenized:
    xy = tokens_to_vector(token, 1)
    data[i, :] = xy
    i += 1    

for token in negative_tokenized:
    xy = tokens_to_vector(token, 0)
    data[i, :] = xy
    i += 1

np.random.shuffle(data)
X = data[:, :-1]
y = data[:, -1]

X_train = X[:-100, ]
y_train = y[:-100, ]

X_test = X[-100:, ]
y_test = y[-100:, ]


model = LogisticRegression()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print("The score is: ", score)

threshold = 0.5
for word, index in iteritems(word_index_map):
    weight = model.coef_[0][index]
    if weight > threshold or weight < -threshold:
        print(word, weight)
