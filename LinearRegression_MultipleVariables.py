import numpy as np
from sklearn.linear_model import LinearRegression

x = [[2,1],[0,1],[1,0],[0,0]]
y = [3,2,3,4]

model = LinearRegression().fit(x, y)

r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
print('wyraz wolny:', model.intercept_)
print('wektor współczynnikow:', model.coef_)
