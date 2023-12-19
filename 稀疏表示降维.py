import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
# Generating synthetic high-dimensional data
np.random.seed(0)
data_dim = 100  # High dimensionality
n_samples = 50

# Generating a dataset with 50 samples in 100 dimensions
X = np.random.randn(n_samples, data_dim)

# Generating sparse representation
# Assuming only 10 dimensions are actually relevant
n_relevant_features = 10
relevant_features = np.random.choice(data_dim, n_relevant_features, replace=False)
X_sparse = np.copy(X)
X_sparse[:, np.setdiff1d(np.arange(data_dim), relevant_features)] = 0

# Plotting
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plotting original high-dimensional data
ax = axes[0]
ax.imshow(X, aspect='auto', cmap='viridis')
ax.set_title("Original High-Dimensional Data")
ax.set_xlabel("Features")
ax.set_ylabel("Samples")

# Plotting sparse representation
ax = axes[1]
ax.imshow(X_sparse, aspect='auto', cmap='viridis')
ax.set_title("Sparse Representation")
ax.set_xlabel("Features")
ax.set_ylabel("Samples")

plt.tight_layout()
plt.show()
