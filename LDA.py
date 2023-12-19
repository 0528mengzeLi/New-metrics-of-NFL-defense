from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')


# Generating two groups of random Gaussian distributed data
np.random.seed(0)
mean1 = [0, 0]
mean2 = [2.5, 2.5]
cov = [[1, 0.8], [0.8, 1]]  # same covariance for both groups
X1 = np.random.multivariate_normal(mean1, cov, 150)
X2 = np.random.multivariate_normal(mean2, cov, 150)

# Combining the data and creating labels
X_combined = np.vstack((X1, X2))
y_combined = np.hstack((np.zeros(X1.shape[0]), np.ones(X2.shape[0])))

# Applying LDA
lda = LDA(n_components=1)
X_lda = lda.fit_transform(X_combined, y_combined)

# Visualizing the original data
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(X1[:, 0], X1[:, 1], alpha=0.7, label='Group 1')
plt.scatter(X2[:, 0], X2[:, 1], alpha=0.7, label='Group 2')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Original Data')
plt.legend()

# Visualizing the data after LDA transformation
plt.subplot(1, 2, 2)
plt.scatter(X_lda[:150], np.zeros(150), alpha=0.7, label='Group 1')
plt.scatter(X_lda[150:], np.zeros(150), alpha=0.7, label='Group 2')
plt.xlabel('LDA Component 1')
plt.title('Data after LDA')
plt.legend()

plt.tight_layout()
plt.show()
