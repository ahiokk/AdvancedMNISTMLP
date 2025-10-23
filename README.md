# AdvancedMNISTMLP

Нейросетевая модель **MLP (Multi-Layer Perceptron)** для распознавания рукописных цифр из датасета **MNIST**, с логированием обучения в **ClearML**.

---

## Возможности
- Использование `nn.ModuleList` для построения кастомной архитектуры  
- Обучение на MNIST с использованием `Adam` и `CrossEntropyLoss`  
- Логирование метрик (loss, accuracy) через **ClearML**  
- Построение графиков потерь и точности  
- Возможность визуализировать результаты обучения в дашборде ClearML  

---

## Технологии
- Python 3.10+
- PyTorch
- Torchvision
- Matplotlib
- ClearML

---

## Структура модели

AdvancedMNISTMLPList(
  (layers): ModuleList(
    (0): Linear(784 → 128)
    (1): ReLU
    (2): Linear(128 → 64)
    (3): ReLU
    (4): Linear(64 → 10)
  )
)


---

## Установка и запуск

```
# Клонируем репозиторий
git clone https://github.com/ahiokk/AdvancedMNISTMLP.git
cd AdvancedMNISTMLP

# Устанавливаем зависимости
pip install -r requirements.txt
```

---

## Обучение

```
python train.py
```

При запуске модель:

1. Загружает MNIST
2. Обучается 10 эпох
3. Отправляет метрики (`loss`, `accuracy`) в ClearML
4. Отображает графики после завершения обучения

---

## Пример вывода

```
Epoch 10/10 — Loss: 0.0451, Accuracy: 98.3%
Test Loss: 0.0412, Test Accuracy: 98.1%
```

---

## ClearML интеграция

Все метрики и графики логируются автоматически.
Для запуска на своём аккаунте ClearML нужно заменить ключи в секции:

```python
Task.set_credentials(
    api_host='https://api.clear.ml',
    web_host='https://app.clear.ml/',
    files_host='https://files.clear.ml',
    key='ВАШ_ACCESS_KEY',
    secret='ВАШ_SECRET_KEY'
)
```

---

## Автор

**ahiok**
Проект создан в рамках обучения MLOps / Deep Learning

```

