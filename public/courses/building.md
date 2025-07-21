# Building Your First Neural Network with PyTorch

Now that your environment is ready, let's build a simple neural network using PyTorch. We'll walk through each part of
the process, from defining the model to making predictions.

---

## What We'll Cover

* Creating synthetic data
* Defining a neural network architecture
* Forward pass and predictions

---

## 1. Create Dummy Data

We'll generate simple input and output tensors to simulate a dataset.

```python
import torch

# Input: 10 samples, each with 3 features
X = torch.rand(10, 3)

# Output: 10 samples, binary labels (0 or 1)
y = torch.randint(0, 2, (10, 1)).float()
```

---

## 2. Define the Neural Network

We'll use `torch.nn` to define a feedforward neural network.

```python
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(3, 5)  # Input layer to hidden layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 1)  # Hidden layer to output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


model = SimpleNet()
print(model)
```

---

## 3. Make Predictions

Now we can do a forward pass to generate predictions from our input data.

```python
with torch.no_grad():
    predictions = model(X)
    print(predictions)
```

> Note: `torch.no_grad()` disables gradient tracking since we're not training yet.

---

## What's Next?

In the next section, we'll:

* Add a loss function and optimizer
* Train the model using backpropagation
* Evaluate its performance

