from sklearn.datasets import load_digits
from sklearn.mixture import GaussianMixture as GMM
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from scipy.stats import mode
from sklearn.model_selection import GridSearchCV

digits = load_digits()

'''Check for numbers of components'''
plt.figure(2)
pca1 = PCA().fit(digits.data)
plt.plot(np.cumsum(pca1.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')


'''make model'''
pca = PCA(n_components=16, random_state=42)

gmm = GMM(n_components=10, random_state=42)
# gmm.fit(digits.data)
model = make_pipeline(pca, gmm)
clusters = model.fit_predict(digits.data)

'''make labels'''
labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]

'''accuracy and results'''
print(accuracy_score(digits.target, labels))

plt.figure(3)
mat = confusion_matrix(digits.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
xticklabels=digits.target_names,
yticklabels=digits.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()