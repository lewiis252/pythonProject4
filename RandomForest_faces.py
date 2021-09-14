from sklearn.datasets import fetch_lfw_people
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import MinMaxScaler
sns.set()


faces = fetch_lfw_people(min_faces_per_person=40)
rand_state = 42
# print(faces.images.shape)

'''Check for numbers of components'''
plt.figure(2)
pca1 = PCA().fit(faces.data)
plt.plot(np.cumsum(pca1.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

'''Model to evaluate'''
Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target, random_state=rand_state)
scaler = MinMaxScaler()

Xtrain = scaler.fit_transform(Xtrain)
Xtest = scaler.fit_transform(Xtest)

pca = PCA(n_components=50, random_state=rand_state)
pca_projected = pca.fit_transform(faces.data)
rfc = RandomForestClassifier(n_estimators=5, random_state=rand_state)

model = make_pipeline(pca, rfc)
print("\nRandom Forest model")
scores = cross_val_score(model, faces.data, faces.target, cv=5)
print("%0.2f test accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
'''Fit model'''
model.fit(Xtrain, ytrain)
print('Train accuracy', model.score(Xtrain, ytrain))
print('Test accuracy', model.score(Xtest, ytest))
yfit = model.predict(Xtest)


'''Report'''
# print(classification_report(ytest, yfit, target_names=faces.target_names))

plt.figure(3)
mat = confusion_matrix(ytest, yfit)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=faces.target_names,
            yticklabels=faces.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')


'''support vector machines model'''
print('\nSuport Vector Machines model')
svc = SVC(kernel='rbf', class_weight='balanced')

model = make_pipeline(pca, svc)

scores = cross_val_score(model, faces.data, faces.target, cv=5)
print("%0.2f test accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
'''Fit model'''
model.fit(Xtrain, ytrain)
print('Train accuracy', model.score(Xtrain, ytrain))
print('Test accuracy', model.score(Xtest, ytest))
yfit = model.predict(Xtest)


'''Report'''
# print(classification_report(ytest, yfit, target_names=faces.target_names))

plt.figure(4)
mat = confusion_matrix(ytest, yfit)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=faces.target_names,
            yticklabels=faces.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')





'''Logistic regression  model'''
print('\nLogistic regression  model')
'''Logistic reg'''
logreg = LogisticRegression(solver='lbfgs', max_iter=10000)
logreg.fit(Xtrain, ytrain)

model = make_pipeline(pca, logreg)

scores = cross_val_score(model, faces.data, faces.target, cv=5)
print("%0.2f test accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
'''Fit model'''
model.fit(Xtrain, ytrain)
print('Train accuracy', model.score(Xtrain, ytrain))
print('Test accuracy', model.score(Xtest, ytest))
yfit = model.predict(Xtest)


'''Report'''
# print(classification_report(ytest, yfit, target_names=faces.target_names))

plt.figure(5)
mat = confusion_matrix(ytest, yfit)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=faces.target_names,
            yticklabels=faces.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')

'''LinearSVC model'''
print('\nLinearSVC  model')

lin_svc = LinearSVC(max_iter=10000)
lin_svc.fit(Xtrain, ytrain)

model = make_pipeline(pca, lin_svc)

scores = cross_val_score(model, faces.data, faces.target, cv=5)
print("%0.2f test accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
'''Fit model'''
model.fit(Xtrain, ytrain)
print('Train accuracy', model.score(Xtrain, ytrain))
print('Test accuracy', model.score(Xtest, ytest))
yfit = model.predict(Xtest)


'''Report'''
# print(classification_report(ytest, yfit, target_names=faces.target_names))

plt.figure(6)
mat = confusion_matrix(ytest, yfit)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=faces.target_names,
            yticklabels=faces.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')


#
# '''Voting clasifier'''
# print('\nVoting clasifier model')
# from sklearn.ensemble import VotingClassifier
# voting_cl = VotingClassifier(
#     estimators = [('rfc', rfc),('svc', svc),('logreg', logreg),('linearsvc', lin_svc)],
#     voting='soft')
# model = voting_cl.fit(Xtrain, ytrain)
#
# scores = cross_val_score(model, faces.data, faces.target, cv=5)
# print("%0.2f test accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
# '''Fit model'''
# model.fit(Xtrain, ytrain)
# print('Train accuracy', model.score(Xtrain, ytrain))
# print('Test accuracy', model.score(Xtest, ytest))
# yfit = model.predict(Xtest)
#
#
# '''Report'''
# # print(classification_report(ytest, yfit, target_names=faces.target_names))
#
# plt.figure(6)
# mat = confusion_matrix(ytest, yfit)
# sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=faces.target_names,
#             yticklabels=faces.target_names)
# plt.xlabel('true label')
# plt.ylabel('predicted label')

plt.show()