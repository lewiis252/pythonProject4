import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

sns.set()

df = pd.read_excel('Lody.xlsx', sheet_name='Lody', index_col='data')

df['Rolada Koral'] = df['Rolada Koral']*10 # manipulacja dla wyglądu danych

print(df.head())



print(df.columns)

column_names = ['śr temp', 'opady (cm)', 'wiatr (km/h)',
       'Miesiąc', 'Pon', 'Wto', 'Śro', 'Czw', 'Pią']
# for i in column_names:
#        sns.scatterplot(data=df, x=i, y="Rolada Koral")

sns.pairplot(df, x_vars=column_names[0:4],
    y_vars='Rolada Koral',
)

X = df[column_names]
y = df['Rolada Koral']

model = LinearRegression(fit_intercept=False)
model.fit(X, y)
df['Przewidywane:regresja liniowa'] = model.predict(X)

df[['Rolada Koral', 'Przewidywane:regresja liniowa']].plot(alpha=0.5)


params = pd.Series(model.coef_, index=X.columns)

from sklearn.utils import resample
np.random.seed(1)
err = np.std([model.fit(*resample(X,y)).coef_ for i in range(1000)],0)

results = pd.DataFrame({'efekt':params.round(0), 'błąd': err.round(0)})
print(results)

'''Random Forest Regression'''
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor()
forest.fit(X,y)

df['Przewidywane: random forest'] = forest.predict(X)
df[['Rolada Koral', 'Przewidywane: random forest']].plot(alpha=0.5)

'''Ridge regression'''

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

# ridge_model = make_pipeline(GaussianFeatures(30),Ridge(alpha=0.1))
# ridge_model.fit(X[:, np.newaxis], y)
#
#
# df['Przewidywane: Ridge'] = ridge_model.predict(X[:, np.newaxis])
# df[['Rolada Koral', 'Przewidywane: Ridge']].plot(alpha=0.5)

plt.show()
