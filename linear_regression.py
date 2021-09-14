import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

'''wszystkie dla 1 zmiennej!!!'''

'''regresja liniowa '''
df1 = pd.read_excel('regresja_dane.xlsx', sheet_name='liniowa', engine='openpyxl')

x = np.array(df1['x']).reshape(((-1,1)))
y = np.array(df1['y'])

model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)

print('a_0:', model.intercept_) #a_0

print('a_1:', model.coef_) #a_1
print('Oszacowany model Y=', model.intercept_, '+', model.coef_, 'x')

'''regresja nieliniowa'''
df1 = pd.read_excel('regresja_dane.xlsx', sheet_name='wykładnicza', engine='openpyxl')

x = np.array(df1['x']).reshape(((-1,1)))
y = np.array(df1['y'])
y = np.log(y)

model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)

a_0 = np.exp(model.intercept_)
print('a_0:', a_0) #a_0

print('a_1:', model.coef_[0]) #a_1
print('Oszacowany model Y=', a_0, 'exp(', model.coef_[0], 't', ')')



