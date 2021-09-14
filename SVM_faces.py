from sklearn.datasets import fetch_lfw_people

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


faces = fetch_lfw_people(min_faces_per_person=40)
rand_state = 42
print(faces.images.shape)

'''Check for numbers of components'''
plt.figure(2)
pca1 = PCA().fit(faces.data)
plt.plot(np.cumsum(pca1.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

'''Model to evaluate'''
Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target, random_state=rand_state, test_size=0.1)
pca = PCA(n_components=250, random_state=rand_state)
pca_projected = pca.fit_transform(faces.data)
svc = SVC(kernel='rbf', class_weight='balanced')

model = make_pipeline(pca, svc)

'''Fit model'''
model.fit(Xtrain, ytrain)
print('Train accuracy', model.score(Xtrain, ytrain))
print('Test accuracy', model.score(Xtest, ytest))
yfit = model.predict(Xtest)


'''Report'''
print(classification_report(ytest, yfit, target_names=faces.target_names))

plt.figure(3)
mat = confusion_matrix(ytest, yfit)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=faces.target_names,
            yticklabels=faces.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')

plt.show()