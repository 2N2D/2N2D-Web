# Training Your Neural Network with PyTorch

In this section, you'll learn how to train the neural network you built. We'll cover the loss function, optimizer,
backpropagation, and running multiple training epochs.

---

## What We'll Cover

* Defining a loss function
* Choosing an optimizer
* Training over multiple epochs

---

## 1. Define the Loss Function

Since this is a binary classification problem, we'll use **Binary Cross-Entropy Loss**.

```python
import torch.nn as nn

criterion = nn.BCELoss()
```

---

## 2. Define the Optimizer

We'll use **Stochastic Gradient Descent (SGD)** to update our model's weights.

```python
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.1)
```

---

## 3. Train the Model

Now let's train the model over multiple epochs.

```python
epochs = 100
for epoch in range(epochs):
    model.train()

    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss occasionally
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
```

---

## 4. Evaluate the Model

Check the model’s predictions after training.

```python
model.eval()
with torch.no_grad():
    predictions = model(X)
    predicted_classes = (predictions > 0.5).float()
    print("Predicted classes:", predicted_classes.view(-1))
    print("Actual labels:", y.view(-1))
```

---

## Summary

You've now:

* Defined a loss function and optimizer
* Trained the model with backpropagation
* Evaluated prediction results


