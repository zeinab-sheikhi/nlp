import numpy as np

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

print(data)
print(type(data))
print(data.keys())
print(data.data)
print(data.data.shape)
print(data.target)
print(data.target_names)
print(data.target.shape)
print(data.feature_names)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33)