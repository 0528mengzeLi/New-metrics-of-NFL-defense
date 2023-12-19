from sklearn.decomposition import SparsePCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
# 生成两个高斯分布的随机数据
mean1, cov1 = [2, 2], [[1, 0.5], [0.5, 1]]  # 第一个高斯分布的均值和协方差矩阵
mean2, cov2 = [-2, -2], [[1, -0.5], [-0.5, 1]]  # 第二个高斯分布的均值和协方差矩阵
data1 = np.random.multivariate_normal(mean1, cov1, 200)
data2 = np.random.multivariate_normal(mean2, cov2, 200)

# 合并数据集并创建标签
X_combined = np.vstack((data1, data2))
y_labels = np.array([0] * len(data1) + [1] * len(data2))

# 应用 Sparse PCA
spca = SparsePCA(n_components=2, random_state=0)
X_spca = spca.fit_transform(X_combined)

# 应用 LDA (作为SLDA的近似)
lda = LDA(n_components=1)  # 使用1个组件，因为我们只有两个类别
X_lda = lda.fit_transform(X_combined, y_labels)
# 可视化 Sparse PCA 的结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_spca[:200, 0], X_spca[:200, 1], alpha=0.7, color='blue', label='Gaussian 1')
plt.scatter(X_spca[200:, 0], X_spca[200:, 1], alpha=0.7, color='red', label='Gaussian 2')
plt.title('Sparse PCA (SPCA)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.legend()

# 可视化 LDA 的结果（SLDA的近似）
plt.subplot(1, 2, 2)
plt.scatter(X_lda[:200, 0], np.zeros(200), alpha=0.7, color='blue', label='Gaussian 1')
plt.scatter(X_lda[200:, 0], np.zeros(200), alpha=0.7, color='red', label='Gaussian 2')
plt.title('Sparse LDA (SLDA) Approximation')
plt.xlabel('LD 1')
plt.yticks([])
plt.legend()

plt.tight_layout()
plt.show()



