# Working with Real-World Data

You've built, trained, and improved a neural network with synthetic data. Now, let’s apply your knowledge to real-world
datasets using PyTorch.

---

## What We'll Cover

* Loading real datasets
* Preprocessing data
* Adapting your network
* Training and evaluation

---

## 1. Load a Real Dataset

We’ll use the classic Iris dataset as an example.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

# Load data
iris = load_iris()
X = iris.data
y = (iris.target == 0).astype(float)  # Binary classification: class 0 vs others

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Create datasets and loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)
```

---

## 2. Adapt the Neural Network

Change the input size to match the dataset (4 features):

```python
class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


model = IrisNet()
```

---

## ️ 3. Train and Evaluate

Use the same training loop and evaluation logic from earlier.

```python
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    model.train()
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Accuracy on real data: {100 * correct / total:.2f}%")
```

---

## Summary

You’ve now:

* Loaded and preprocessed real-world data
* Built and trained a PyTorch model
* Measured accuracy on test data

Congratulations! You've completed your first full neural network learning pipeline—from fundamentals to real-world
application.
