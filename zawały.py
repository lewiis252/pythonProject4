import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel('dane.xlsx', sheet_name='zawały')
df = df.dropna()
df = df.reset_index(drop=True)
# df = df.fillna(df.mean())

df = df.drop(columns=['Czas choroby wiencowej', 'Rodzaj zawału'])

df = pd.get_dummies(df)

# print(df.columns)

features = df.loc[:, 'Wiek':'Palenie_pali']

x = features.values
y = df["Zawał_Tak"].values

# print((df.to_string()))

x_train, x_test, y_train, y_test = train_test_split(x, y)

# print(x_test)
'''Random Forest'''
model = RandomForestClassifier(1000)
model.fit(x_train, y_train)
# print(model.predict(x_test))
print('Random forest test score: ', model.score(x_test, y_test))
# print(x_test)
# print(model.predict(x_test))

'''Logistic reg'''
logreg = LogisticRegression(solver='lbfgs',max_iter=10000)
logreg.fit(x_train, y_train)
print('Logistic regression test score: ', logreg.score(x_test, y_test))
# print(logreg.predict(x_test))

'''k_NN'''
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=2)
clf.fit(x_train, y_train)
print('kNN test score: ', clf.score(x_test, y_test))
# print(clf.predict(x_test))

'''LinearSVC'''
from sklearn.svm import LinearSVC
lin_svc = LinearSVC(max_iter=10000)
lin_svc.fit(x_train, y_train)
print('Linear SVC test score: ', clf.score(x_test, y_test))
# print(lin_svc.predict(x_test))

'''gradient boosted tree'''
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(max_depth=50)
gbc.fit(x_train, y_train)
print('Gadient boosting classifier test score: ', gbc.score(x_test, y_test))
# print(gbc.predict(x_test))


'''PCA'''
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(x)
x_pca = pca.transform(x)
X_pca = pd.DataFrame(x_pca, columns=['first component', 'second component'])

plt.figure(1)
sns.heatmap(pca.components_, cmap="YlGnBu")

plt.figure(2)
sns.scatterplot(data=X_pca, x='first component', y='second component', hue = y)
# pca_result = pd.DataFrame(d)
plt.show()


