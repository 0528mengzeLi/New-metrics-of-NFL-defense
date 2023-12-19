import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import numpy as np
import glob
from sklearn.decomposition import KernelPCA
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
# 应用Kernel PCA
kernel_pca = KernelPCA(n_components=2, kernel='poly', degree=3, coef0=1, gamma=10)
X_kpca_poly = kernel_pca.fit_transform(images)

# 可视化
colors = cm.rainbow(np.linspace(0, 1, len(np.unique(labels))))
plt.figure(figsize=(12, 10))
for label, color in zip(np.unique(labels), colors):
    indices = np.where(labels == label)
    plt.scatter(X_kpca_poly[indices, 0], X_kpca_poly[indices, 1], label=label, color=color, s=50)
plt.title('Kernel PCA of Yale Face Database (Polynomial Kernel)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

