
import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import DataLoader

from torchvision import datasets, transforms

import matplotlib.pyplot as plt

from clearml import Task, Logger


# Определяем модель AdvancedMNISTMLPList
class AdvancedMNISTMLPList(nn.Module):
    def __init__(self):
        super(AdvancedMNISTMLPList, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        ])
    def forward(self, x):
        x = x.view(-1, 28*28)  # раскладываем картинку в вектор
        for layer in self.layers:
            x = layer(x)
        return x


model = AdvancedMNISTMLPList()


# Загружаем данные MNIST (тренировочный и тестовый набор)
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Инициализируем функцию потерь и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



# Подключение к ClearML

Task.set_credentials(
    api_host='https://api.clear.ml',
    web_host='https://app.clear.ml/',
    files_host='https://files.clear.ml',
    key='ВАШ_ACCESS_KEY',
    secret='ВАШ_SECRET_KEY'
)
task = Task.init(project_name='MNIST_Project', task_name='MLP_Train')


# Цикл обучения


num_epochs = 10
train_losses = []
train_accuracies = []


for epoch in range(num_epochs):
    # Перевод модели в режим обучения
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0


    # Проход по всем батчам трен.набора
    for images, labels in train_loader:

        
        # Прямой проход
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Обратный проход и оптимизация
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # Накопление статистики для метрик
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()


    # Средние значения по эпохе
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)


    # Выводим информацию по эпохе
    print(f'Epoch {epoch+1}/{num_epochs} — Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    
    # Отправляем метрики в ClearML
    # Отправляем loss
    Logger.current_logger().report_scalar('train', 'loss', iteration=epoch, value=epoch_loss)
    # Отправляем accuracy
    Logger.current_logger().report_scalar('train', 'accuracy', iteration=epoch, value=epoch_acc)


# Оценка на тестовом наборе
# Перевод модели в режим оценки
model.eval()


test_loss = 0.0

correct = 0

total = 0

all_preds = []
all_labels = []


with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


test_loss = test_loss / len(test_loader)
test_accuracy = 100.0 * correct / total

print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')


# Построение графиков потерь и точности
epochs = range(1, num_epochs+1)
plt.figure(figsize=(12,5))


plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, marker='o', color='blue', label='Train Loss')
plt.title('Loss vs Epochs')
plt.xlabel('Эпоха')
plt.ylabel('Потеря')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, marker='o', color='green', label='Train Accuracy')
plt.title('Accuracy vs Epochs')
plt.xlabel('Эпоха')
plt.ylabel('Точность (%)')
plt.legend()


plt.tight_layout()
plt.show()