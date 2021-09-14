from sklearn.datasets import load_digits
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
sns.set()


digits = load_digits()


'''2D visualise'''
pca = PCA(n_components=2, random_state=42)
pca_projected = pca.fit_transform(digits.data)
pca_projected = pd.DataFrame(pca_projected)

hue = digits.target

plt.figure(1)
sns.scatterplot(data=pca_projected, x=0, y=1, hue=hue, palette='hls')

'''Check for numbers of components'''
plt.figure(2)
pca1 = PCA().fit(digits.data)
plt.plot(np.cumsum(pca1.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

'''Model to evaluate'''
Xtrain, Xtest, ytrain, ytest = train_test_split(digits.data, digits.target, random_state=42)
pca = PCA(n_components=20, random_state=42)
pca_projected = pca.fit_transform(digits.data)
svc = SVC(kernel='rbf', class_weight='balanced')

model = make_pipeline(pca, svc)

'''Fit model'''
model.fit(Xtrain, ytrain)
print('Train accuracy', model.score(Xtrain, ytrain))
print('Test accuracy', model.score(Xtest, ytest))
yfit = model.predict(Xtest)


'''Report'''
print(classification_report(ytest, yfit))

plt.figure(3)
mat = confusion_matrix(ytest, yfit)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')

plt.show()