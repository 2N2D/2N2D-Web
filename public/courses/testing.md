# Testing Your Neural Network

Training a model on the same data it’s evaluated on doesn’t give a full picture of its performance. In this section,
you’ll learn how to split your data into training and testing sets to better understand generalization.

---

## What We'll Cover

* Splitting data into training and test sets
* Training on training data only
* Evaluating on unseen test data

---

## 1. Split the Data

We’ll use `torch.utils.data.random_split` to divide our dataset.

```python
from torch.utils.data import TensorDataset, random_split, DataLoader

# Combine inputs and labels into a dataset
full_dataset = TensorDataset(X, y)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2)
```

---

## ️ 2. Train on the Training Set

Modify your training loop to iterate over `train_loader`:

```python
epochs = 100
for epoch in range(epochs):
    model.train()
    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
```

---

## 3. Evaluate on the Test Set

Now evaluate how well the model performs on unseen data:

```python
model.eval()
total = 0
correct = 0

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        predicted = (outputs > 0.5).float()
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
```

---

## Summary

You’ve now:

* Split your data into training and testing sets
* Trained only on training data
* Evaluated the model on unseen data


