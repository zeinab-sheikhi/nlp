import nltk
import random

from bs4 import BeautifulSoup


positive_reviews = BeautifulSoup(open('./data/electronics/positive.review').read(), features="html.parser")
positive_reviews = positive_reviews.findAll('review_text')

trigrams = {}
trigrams_probabilities = {}


for review in positive_reviews:
    s = review.text.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    for i in range(len(tokens) - 2):
        k = (tokens[i], tokens[i + 2])
        if k not in trigrams:
            trigrams[k] = []
        trigrams[k].append(tokens[i + 1])


for k, words in trigrams.items():
    if len(set(words)) > 1:
        new_dict = {}
        for word in words:
            if word not in new_dict:
                new_dict[word] = 0
            new_dict[word] += 1
            for word, count in new_dict.items():
                new_dict[word] = float(count) / len(words)
        trigrams_probabilities[k] = new_dict


def random_sample(d):
    r = random.random()
    cumulative = 0
    for word, probability in d.items():
        cumulative += probability
        if r < cumulative:
            return word


def test_spinner():
    review = random.choice(positive_reviews)
    s = review.text.lower()
    print("Original", s)
    tokens = nltk.tokenize.word_tokenize(s)
    for i in range(len(tokens) - 2):
        if random.random() < 0.2:
            k = (tokens[i], tokens[i + 2])
            if k in trigrams_probabilities:
                w = random_sample(trigrams_probabilities[k])
                tokens[i + 1] = w
    print("Spun") 
    s = ""
    for t in tokens:
        if t is not None:
            s += t + " "
    print(s)


test_spinner()