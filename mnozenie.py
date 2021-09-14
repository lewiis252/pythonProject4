import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel('dane.xlsx', sheet_name='mnozenie')
x = df[['x1','x2']]
y = df['y']
x_train, x_test, y_train, y_test = train_test_split(x, y)

sns.pairplot(df)
plt.show()

# model = RandomForestRegressor()
# model.fit(x_train,y_train)
# print(model.score(x_train, y_train))
# print(model.score(x_test, y_test))

model2 = LinearRegression()
model2.fit(x_train,y_train)
print(model2.score(x_train, y_train))
print(model2.score(x_test, y_test))

print(model2.predict([[100,5]]))
print(model2.predict([[0.5, 0.5]]))


scaler = Normalizer()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
model2.fit(x_train,y_train)
print(model2.score(x_train, y_train))
print(model2.score(x_test, y_test))


# print(model.predict([[10, 5]]))
# print(model.predict([[0.5, 0.5]]))
print(model2.predict([[100,5]]))
print(model2.predict([[0.5, 0.5]]))