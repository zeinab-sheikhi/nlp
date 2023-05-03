import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud

df = pd.read_csv(r"D:\Mine\Udemy\nlp\nlp\spam.csv",  encoding="ISO_8859_1")
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
df.columns = ['labels', 'data']
df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})

y = df['b_labels'].to_numpy()

count_vectorizer = CountVectorizer(decode_error='ignore')
X = count_vectorizer.fit_transform(df['data'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

model = MultinomialNB()
model.fit(X_train, y_train)
print("Train score:", model.score(X_train, y_train))
print("Test score is:", model.score(X_test, y_test))


def visualize(label):
    words = ''
    for msg in df[df['labels'] == label]['data']:
        msg = msg.lower()
        words += msg + ' '
    wordcloud = WordCloud(width=600, height=400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()   


visualize('spam')
visualize('ham')

df['predictions'] = model.predict(X)

# Things that should be spam
sneaky_spam = df[(df['predictions'] == 0)  & (df['b_labels'] == 1)]['data']
print("Things that should be spam")
for msg in sneaky_spam:
    print(msg)

# Things that should not be spamed
not_actually_spam = df[(df['predictions'] == 1) & (df['b_labels'] == 0)]['data']
print('Things that should not be spamed')
for msg in not_actually_spam: 
    print(msg)