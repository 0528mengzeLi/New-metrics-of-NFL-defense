import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('TkAgg')
# Generating random Gaussian distributed data
np.random.seed(0)
mean = [0, 0]
cov = [[1, 0.8], [0.8, 1]]  # covariance matrix
X = np.random.multivariate_normal(mean, cov, 300)

# Applying PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualizing the original data
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.7)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Original Data')

# Visualizing the data after PCA transformation
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Data after PCA')

# Drawing the principal axes
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    arrowprops=dict(arrowstyle='->',linewidth=2,shrinkA=0, shrinkB=0)
    plt.annotate('', pca.mean_ + v, pca.mean_, arrowprops=arrowprops)

plt.tight_layout()
plt.show()
