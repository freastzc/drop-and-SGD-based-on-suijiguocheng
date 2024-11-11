import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 加载MNIST数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)


# 定义简单CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


model = SimpleCNN()

# 固定学习率设置
initial_lr = 0.01
optimizer = optim.Adam(model.parameters(), lr=initial_lr)
criterion = nn.CrossEntropyLoss()

# 训练过程
def train_fixed_lr(model, optimizer, criterion, num_epochs=10):
    model.train()
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(train_loader))
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")
    return losses

losses_fixed_lr = train_fixed_lr(model, optimizer, criterion)




# 定义随机过程调节学习率
class MarkovLearningRateScheduler:
    def __init__(self, initial_lr, high_lr=0.01, low_lr=0.001):
        self.current_lr = initial_lr
        self.high_lr = high_lr
        self.low_lr = low_lr

    def update_lr(self, epoch_loss):
        # 简单的条件判断：如果损失增加，降低学习率；如果损失减少，提升学习率
        if epoch_loss > 0.02:
            self.current_lr = max(self.low_lr, self.current_lr * 0.9)
        else:
            self.current_lr = min(self.high_lr, self.current_lr * 1.1)
        return self.current_lr


# 动态调整学习率的训练过程
def train_dynamic_lr(model, criterion, num_epochs=10):
    scheduler = MarkovLearningRateScheduler(initial_lr)
    optimizer = optim.Adam(model.parameters(), lr=scheduler.current_lr)
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # 更新学习率
        current_lr = scheduler.update_lr(epoch_loss / len(train_loader))
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        losses.append(epoch_loss / len(train_loader))
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}, LR: {current_lr:.5f}")
    return losses


# 执行动态学习率训练
losses_dynamic_lr = train_dynamic_lr(model, criterion)


# 绘制损失曲线
import matplotlib.pyplot as plt


# 保存训练和验证损失的函数
def train_and_validate(model, optimizer, criterion, num_epochs=10, dynamic_lr=False):
    train_losses, val_losses, learning_rates = [], [], []

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        # 记录训练损失
        train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        epoch_val_loss = 0
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # 记录验证损失
        val_loss = epoch_val_loss / len(test_loader)
        val_losses.append(val_loss)

        # 记录学习率
        if dynamic_lr:
            current_lr = scheduler.update_lr(train_loss)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            learning_rates.append(current_lr)
        else:
            learning_rates.append(optimizer.param_groups[0]['lr'])

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {learning_rates[-1]:.5f}")

    final_accuracy = correct / total
    return train_losses, val_losses, learning_rates, final_accuracy


# 初始化模型和优化器
model_fixed_lr = SimpleCNN()
model_dynamic_lr = SimpleCNN()
optimizer_fixed = optim.Adam(model_fixed_lr.parameters(), lr=initial_lr)
optimizer_dynamic = optim.Adam(model_dynamic_lr.parameters(), lr=initial_lr)
scheduler = MarkovLearningRateScheduler(initial_lr)

# 运行初始实验和动态学习率实验
train_losses_fixed, val_losses_fixed, learning_rates_fixed, final_accuracy_fixed = train_and_validate(
    model_fixed_lr, optimizer_fixed, criterion, dynamic_lr=False)

train_losses_dynamic, val_losses_dynamic, learning_rates_dynamic, final_accuracy_dynamic = train_and_validate(
    model_dynamic_lr, optimizer_dynamic, criterion, dynamic_lr=True)

# 绘制训练和验证损失曲线
plt.figure(figsize=(14, 5))

# 训练损失
plt.subplot(1, 3, 1)
plt.plot(train_losses_fixed, label="Fixed LR - Train Loss")
plt.plot(train_losses_dynamic, label="Dynamic LR - Train Loss")
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.title("Training Loss Comparison")
plt.legend()

# 验证损失
plt.subplot(1, 3, 2)
plt.plot(val_losses_fixed, label="Fixed LR - Val Loss")
plt.plot(val_losses_dynamic, label="Dynamic LR - Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Validation Loss")
plt.title("Validation Loss Comparison")
plt.legend()

# 学习率变化趋势
plt.subplot(1, 3, 3)
plt.plot(learning_rates_fixed, label="Fixed LR")
plt.plot(learning_rates_dynamic, label="Dynamic LR (Markov)")
plt.xlabel("Epochs")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Trend")
plt.legend()

plt.tight_layout()
plt.show()

# 输出最终精度对比
print(f"Final Accuracy - Fixed LR: {final_accuracy_fixed:.4f}")
print(f"Final Accuracy - Dynamic LR: {final_accuracy_dynamic:.4f}")

