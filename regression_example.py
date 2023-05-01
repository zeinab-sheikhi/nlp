import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('/Users/macbookpro/Python/Udemy/airfoil_self_noise.dat', sep='\t', header=None)
df.head()
df.info()

data = df[[0, 1, 2, 3, 4]].values
target = df[5].values

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33)

model = LinearRegression()
model.fit(X_train, y_train)

# These scores don't represent accuracy because we're not using classification
# These scores are R square, and the closer to 1, the better the model

print(model.score(X_train, y_train))
print(model.score(X_test, y_test))

predictions = model.predict(X_test)
print(predictions)

model2 = RandomForestRegressor()
model2.fit(X_train, y_train)

print(model2.score(X_train, y_train))
print(model2.score(X_test, y_test))


