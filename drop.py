import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 数据加载和增强
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)


# CNN 模型架构
class SimpleCNNWithDropout(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(SimpleCNNWithDropout, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout3 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.dropout1(x)
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = self.dropout2(x)
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout3(x)
        return self.fc2(x)


# 随机 Dropout 概率调度器
class DynamicDropoutScheduler:
    def __init__(self, initial_prob=0.5, high_prob=0.9, low_prob=0.2):
        self.current_prob = initial_prob
        self.high_prob = high_prob
        self.low_prob = low_prob
        self.dropout_rates = [initial_prob]

    def update_prob(self, epoch_loss):
        if epoch_loss > 0.02:
            self.current_prob = max(self.low_prob, self.current_prob * 0.9)
        else:
            self.current_prob = min(self.high_prob, self.current_prob * 1.1)
        self.dropout_rates.append(self.current_prob)
        return self.current_prob


# 训练和验证函数
def train_and_validate(model, optimizer, criterion, num_epochs=10, dynamic_dropout=False):
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    dropout_scheduler = DynamicDropoutScheduler() if dynamic_dropout else None

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss, correct_train = 0, 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            correct_train += (outputs.argmax(1) == labels).sum().item()

        train_losses.append(epoch_train_loss / len(train_loader))
        train_accuracies.append(correct_train / len(train_dataset))

        # 更新丢弃率
        if dynamic_dropout and dropout_scheduler:
            current_prob = dropout_scheduler.update_prob(epoch_train_loss / len(train_loader))
            model.dropout1.p = model.dropout2.p = model.dropout3.p = current_prob

        # 验证模型
        model.eval()
        epoch_val_loss, correct_val = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()
                correct_val += (outputs.argmax(1) == labels).sum().item()

        val_losses.append(epoch_val_loss / len(test_loader))
        val_accuracies.append(correct_val / len(test_dataset))
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, "
              f"Val Loss: {val_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}, "
              f"Val Acc: {val_accuracies[-1]:.4f}")

    return train_losses, val_losses, train_accuracies, val_accuracies, dropout_scheduler.dropout_rates if dynamic_dropout else None


# 训练传统固定 Dropout 模型
model_fixed = SimpleCNNWithDropout(dropout_prob=0.5)
optimizer_fixed = optim.Adam(model_fixed.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()
train_losses_fixed, val_losses_fixed, train_accs_fixed, val_accs_fixed, _ = train_and_validate(model_fixed,
                                                                                               optimizer_fixed,
                                                                                               criterion, num_epochs=10,
                                                                                               dynamic_dropout=False)

# 训练动态丢弃率的贝叶斯 Dropout 模型
model_dynamic = SimpleCNNWithDropout()
optimizer_dynamic = optim.Adam(model_dynamic.parameters(), lr=0.001)
train_losses_dynamic, val_losses_dynamic, train_accs_dynamic, val_accs_dynamic, dropout_rates = train_and_validate(
    model_dynamic, optimizer_dynamic, criterion, num_epochs=10, dynamic_dropout=True)

# 可视化结果
plt.figure(figsize=(15, 10))

# 训练损失对比
plt.subplot(2, 3, 1)
plt.plot(train_losses_fixed, label="Fixed Dropout - Train Loss")
plt.plot(train_losses_dynamic, label="Dynamic Dropout - Train Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# 验证损失对比
plt.subplot(2, 3, 2)
plt.plot(val_losses_fixed, label="Fixed Dropout - Val Loss")
plt.plot(val_losses_dynamic, label="Dynamic Dropout - Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# 训练准确率对比
plt.subplot(2, 3, 3)
plt.plot(train_accs_fixed, label="Fixed Dropout - Train Accuracy")
plt.plot(train_accs_dynamic, label="Dynamic Dropout - Train Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# 验证准确率对比
plt.subplot(2, 3, 4)
plt.plot(val_accs_fixed, label="Fixed Dropout - Val Accuracy")
plt.plot(val_accs_dynamic, label="Dynamic Dropout - Val Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# 动态丢弃率对比
plt.subplot(2, 3, 5)
plt.plot(dropout_rates, label="Dynamic Dropout Rate", color="purple")
plt.axhline(y=0.5, color="orange", linestyle="--", label="Fixed Dropout Rate")
plt.xlabel("Epochs")
plt.ylabel("Dropout Rate")
plt.legend()

plt.tight_layout()
plt.show()
