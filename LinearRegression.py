import numpy as np
from sklearn.linear_model import LinearRegression


x = np.array([1,2,3,4,5,6,7]).reshape((-1, 1))
y = np.array([8,13,14,17,18,19,20])

model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)

print('a_0:', model.intercept_) #a_0

print('a_1:', model.coef_) #a_1
