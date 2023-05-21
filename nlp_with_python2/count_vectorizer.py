import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


vectorizer = CountVectorizer(analyzer="word")
vectorizer = CountVectorizer(analyzer="char")
Xtrain = vectorizer.fit_transform()

Xtest = vectorizer.transform()