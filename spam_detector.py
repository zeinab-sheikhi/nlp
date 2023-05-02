from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np

"""
data = pd.read_csv('spambase.data').as_matrix()
as_matrix() method is used to convert the data frame to a numpy array.
this method is deprecated so use to_numpy() instead.
"""

data = pd.read_csv('spambase.data').to_numpy()

# shuffle method changed the rows' places
np.random.shuffle(data)

# the first 48 columns contain the features
X = data[:, :48]

# the last column indicates whether an email is spam or not
Y = data[:, -1]

X_train = X[:-100, ]
Y_train = Y[:-100, ]

X_test = X[-100:, ]
Y_test = Y[-100:, ]

nb_model = MultinomialNB()
nb_model.fit(X_train, Y_train)

nb_score = nb_model.score(X_test, Y_test)
print("Naive Bayes score is:", nb_score)

ada_model = AdaBoostClassifier()
ada_model.fit(X_train, Y_train)

ada_score = ada_model.score(X_test, Y_test)
print("AdaBoost score is:", ada_score)
