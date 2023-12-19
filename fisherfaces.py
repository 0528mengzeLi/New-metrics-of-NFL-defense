import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib
matplotlib.use('TkAgg')
# Generating synthetic data to simulate different facial conditions
# Class 1: Normal expression, Class 2: Expression change, Class 3: Lighting change, Class 4: Occlusion
np.random.seed(0)
normal_expression = np.random.normal(0, 1, (20, 2)) + np.array([0, 5])
expression_change = np.random.normal(0, 1, (20, 2)) + np.array([2, 4])
lighting_change = np.random.normal(0, 1, (20, 2)) + np.array([4, 3])
occlusion = np.random.normal(0, 1, (20, 2)) + np.array([6, 2])

# Combining the data
X = np.vstack([normal_expression, expression_change, lighting_change, occlusion])
y = np.array([1]*20 + [2]*20 + [3]*20 + [4]*20)

# Applying LDA
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X, y)

# Plotting
plt.figure(figsize=(14, 6))

# Plot before LDA
plt.subplot(1, 2, 1)
plt.scatter(normal_expression[:, 0], normal_expression[:, 1], color='blue', label='Normal Expression')
plt.scatter(expression_change[:, 0], expression_change[:, 1], color='green', label='Expression Change')
plt.scatter(lighting_change[:, 0], lighting_change[:, 1], color='red', label='Lighting Change')
plt.scatter(occlusion[:, 0], occlusion[:, 1], color='purple', label='Occlusion')
plt.title("Before LDA")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

# Plot after LDA
plt.subplot(1, 2, 2)
plt.scatter(X_lda[y == 1][:, 0], X_lda[y == 1][:, 1], color='blue', label='Normal Expression')
plt.scatter(X_lda[y == 2][:, 0], X_lda[y == 2][:, 1], color='green', label='Expression Change')
plt.scatter(X_lda[y == 3][:, 0], X_lda[y == 3][:, 1], color='red', label='Lighting Change')
plt.scatter(X_lda[y == 4][:, 0], X_lda[y == 4][:, 1], color='purple', label='Occlusion')
plt.title("After LDA (Fisherfaces)")
plt.xlabel("LDA Feature 1")
plt.ylabel("LDA Feature 2")
plt.legend()

plt.show()
