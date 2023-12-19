import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
import numpy as np
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# 数据加载和预处理
def load_and_preprocess_images(directory):
    image_paths = glob.glob(directory + '/subject*')
    images = []
    labels = []

    for path in image_paths:
        image = Image.open(path).convert('L')
        resized_image = image.resize((64, 64))
        normalized_image = np.array(resized_image) / 255.0
        images.append(normalized_image)
        label = path.split('/')[-1].split('.')[0]
        labels.append(label)

    return np.array(images), np.array(labels)


nested_directory = 'G:/pycharm/PyCharm Projects/Pattern recognition/big hw/yale-face-database'
images, labels = load_and_preprocess_images(nested_directory)

# 将标签转换为整数
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

# 将numpy数组转换为PyTorch张量
X_train = torch.tensor(X_train).unsqueeze(1).float()  # 增加一个通道维度
X_test = torch.tensor(X_test).unsqueeze(1).float()
y_train = torch.tensor(y_train).long()
y_test = torch.tensor(y_test).long()

# 创建数据加载器
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)


# 构建CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # 计算卷积层输出的尺寸
        self.fc1 = nn.Linear(12544, 128)  # 请根据您的实际输出调整这里的数字
        self.fc2 = nn.Linear(128, len(np.unique(labels_encoded)))

    def forward(self, x):
        # 卷积层和池化层
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        # 展平操作
        x = torch.flatten(x, 1)
        # 全连接层
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x



model = CNN()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # 测试模型
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f'Epoch {epoch + 1}, Accuracy: {100 * correct / total}%')

# 保存模型
torch.save(model.state_dict(), 'cnn_model.pth')
