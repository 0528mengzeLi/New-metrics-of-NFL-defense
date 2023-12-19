import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import glob
import matplotlib
matplotlib.use('TkAgg')
# Function to display a grid of images
def display_image_grid(images, titles, rows, cols, figsize):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    for i, ax in enumerate(axes.flatten()):
        if i < len(images):
            ax.imshow(images[i], cmap='gray')
            ax.set_title(titles[i], fontsize=8)
            ax.axis('off')
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()

# 收集所有图像路径
nested_directory = 'G:/pycharm/PyCharm Projects/Pattern recognition/big hw/yale-face-database'  # 替换为实际解压目录的路径
image_paths = glob.glob(nested_directory + '/subject*')
# Select a subset of images for visualization
sample_images = image_paths[:12]  # Display the first 12 images
sample_titles = [path.split('\\')[-1] for path in sample_images]
loaded_images = [Image.open(path).convert('L') for path in sample_images]

# Display the images before preprocessing
display_image_grid(loaded_images, sample_titles, 3, 4, (12, 6))

# Data Preprocessing (convert to grayscale, resize, normalize)
processed_images = [np.array(img.resize((64, 64))) / 255.0 for img in loaded_images]

# Display the images after preprocessing
display_image_grid(processed_images, sample_titles, 3, 4, (12, 6))

