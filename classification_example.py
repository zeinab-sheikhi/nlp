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
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33)

model = RandomForestClassifier()
model.fit(X_train, y_train)
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))

predictions = model.predict(X_test)
print(predictions)
print(model.score(X_test, predictions))

N = len(y_test)
manual_score = np.sum(predictions == y_test) / N
print(manual_score)

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train2 = scaler.fit_transform(X_train)
X_test2 = scaler.transform(X_test)

model2 = MLPClassifier(max_iter=500)
model2.fit(X_train2, y_train)
print(model.score(X_train2, y_train))
print(model.score(X_test2, y_test))