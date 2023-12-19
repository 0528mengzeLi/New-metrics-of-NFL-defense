from PIL import Image
import numpy as np
import glob
import matplotlib
matplotlib.use('TkAgg')
# 预处理函数
def preprocess_images(image_paths):
    processed_images = []
    labels = []

    for image_path in image_paths:
        # 加载图像
        image = Image.open(image_path)

        # 转换为灰度图像
        gray_image = image.convert('L')

        # 调整大小至64x64
        resized_image = gray_image.resize((64, 64))

        # 归一化像素值
        normalized_image = np.array(resized_image) / 255.0

        # 将图像展平为一维数组
        flat_image = normalized_image.flatten()

        processed_images.append(flat_image)

        # 从文件名中提取标签（例如，subject01, subject02等）
        label = image_path.split('/')[-1].split('.')[0]
        labels.append(label)

    return np.array(processed_images), np.array(labels)

# 收集所有图像路径
nested_directory = 'G:/pycharm/PyCharm Projects/Pattern recognition/big hw/yale-face-database'  # 替换为实际解压目录的路径
image_paths = glob.glob(nested_directory + '/subject*')

# 预处理图像
images, labels = preprocess_images(image_paths)

from sklearn.model_selection import train_test_split

# 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)


from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

pca_components = [20, 40, 60]
pca_results = {}

for n_components in pca_components:
    # 应用PCA
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    # k-最近邻分类器，k=3
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_pca, y_train)
    y_pred = knn.predict(X_test_pca)

    # 评估并存储结果
    accuracy = accuracy_score(y_test, y_pred)
    pca_results[n_components] = accuracy


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# LDA设置
lda_dim = 14
lda = LDA(n_components=lda_dim)
lda.fit(X_train, y_train)
X_train_lda = lda.transform(X_train)
X_test_lda = lda.transform(X_test)

# k-最近邻分类器，k=3，用于LDA
knn_lda = KNeighborsClassifier(n_neighbors=3)
knn_lda.fit(X_train_lda, y_train)
y_pred_lda = knn_lda.predict(X_test_lda)

# 评估LDA结果
lda_accuracy = accuracy_score(y_test, y_pred_lda)


# PCA和LDA的准确率结果
print(pca_results)
print(lda_accuracy)


import matplotlib.pyplot as plt
# Visualization of PCA and LDA results

# Plotting PCA accuracies
plt.figure(figsize=(10, 5))
plt.bar(range(len(pca_results)), list(pca_results.values()), align='center')
plt.xticks(range(len(pca_results)), list(pca_results.keys()))
plt.xlabel('Number of Principal Components')
plt.ylabel('Accuracy')
plt.title('PCA Accuracy with Different Number of Components')
plt.show()

# Visualization of LDA result
plt.figure(figsize=(5, 3))
plt.bar('LDA', lda_accuracy)
plt.xlabel('Method')
plt.ylabel('Accuracy')
plt.title('LDA Accuracy')
plt.ylim(0, 1)
plt.show()

from sklearn.decomposition import PCA


# 特征脸绘图函数
def plot_eigenfaces(pca_components, title):
    pca = PCA(n_components=pca_components)
    pca.fit(X_train)
    eigenfaces = pca.components_.reshape((pca_components, 64, 64))

    cols = min(pca_components, 10)
    rows = pca_components // cols + (pca_components % cols > 0)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 1.5 * rows))

    for i, ax in enumerate(axes.flatten()):
        if i < pca_components:
            ax.imshow(eigenfaces[i], cmap='gray')
            ax.axis('off')
        else:
            ax.axis('off')

    plt.suptitle(title, fontsize=16)
    plt.show()

# 使用20个PCA主成分的特征脸
plot_eigenfaces(20, 'Eigenfaces with 20 PCA Components')

# 使用40个PCA主成分的特征脸
plot_eigenfaces(40, 'Eigenfaces with 40 PCA Components')

# 使用60个PCA主成分的特征脸
plot_eigenfaces(60, 'Eigenfaces with 60 PCA Components')

# Fisherfaces Visualization for LDA
lda = LDA(n_components=14)
lda.fit(X_train, y_train)
fisherfaces = lda.scalings_.T.reshape((14, 64, 64))

fig, axes = plt.subplots(2, 7, figsize=(15, 4))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(fisherfaces[i], cmap='gray')
    ax.axis('off')
plt.suptitle('Fisherfaces (LDA Components)', fontsize=16)
plt.show()

