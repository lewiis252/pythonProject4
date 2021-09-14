import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.decomposition import PCA

from sklearn.datasets import load_digits
# digits are in 64 dimension
digits = load_digits()

pca = PCA(2) # project from 64 to 2 dimensions
projected = pca.fit_transform(digits.data)
print(digits.data.shape)
print(projected.shape)


# plt.scatter(projected[:, 0], projected[:, 1],c=digits.target, edgecolor='none', alpha=0.5,
#             cmap=plt.cm.get_cmap('Accent', 10))
# plt.xlabel('component 1')
# plt.ylabel('component 2')
# plt.colorbar()


# optimal number of components
pca = PCA().fit(digits.data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')


plt.show()