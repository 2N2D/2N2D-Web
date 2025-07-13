# PyTorch Neural Network Cheat Sheet

This cheat sheet summarizes essential PyTorch components used in building and training neural networks, along with their
purpose and common use cases.

---

## Imports

| Function                                                 | Description                              | Use When                      |
|----------------------------------------------------------|------------------------------------------|-------------------------------|
| `import torch`                                           | Core PyTorch library                     | Always                        |
| `import torch.nn as nn`                                  | Neural network layers and loss functions | Defining models, losses       |
| `import torch.optim as optim`                            | Optimizers for training                  | Training a model              |
| `from torch.utils.data import DataLoader, TensorDataset` | Batching and managing datasets           | Training/testing with batches |

---

## Data Handling

| Function                                         | Description                  | Use When                       |
|--------------------------------------------------|------------------------------|--------------------------------|
| `TensorDataset(X, y)`                            | Wraps tensors into a dataset | Preparing for DataLoader       |
| `DataLoader(dataset, batch_size, shuffle=True)`  | Loads batches of data        | Training/testing               |
| `random_split(dataset, [train_size, test_size])` | Splits a dataset             | Creating training/testing sets |

---

## Tensors

| Function                                  | Description                       | Use When                           |
|-------------------------------------------|-----------------------------------|------------------------------------|
| `torch.tensor(data, dtype=torch.float32)` | Creates a tensor                  | Converting from NumPy or lists     |
| `torch.rand(size)`                        | Random tensor with uniform values | Dummy input features               |
| `torch.randint(low, high, size)`          | Random integer tensor             | Dummy labels                       |
| `.float()`                                | Converts tensor to float type     | For compatibility in models/losses |
| `.unsqueeze(dim)`                         | Adds a dimension                  | When shape mismatch occurs         |

---

## Neural Network

| Function                               | Description                                 | Use When                   |
|----------------------------------------|---------------------------------------------|----------------------------|
| `nn.Linear(in_features, out_features)` | Fully connected layer                       | Creating layers in a model |
| `nn.ReLU()`                            | Activation function                         | Non-linearity after layers |
| `nn.Sigmoid()`                         | Activation for binary classification output | Final output layer (0-1)   |
| `model = YourModelClass()`             | Instantiates your model                     | Before training            |
| `model(x)`                             | Forward pass                                | Producing predictions      |

---

## Training

| Function                                | Description                 | Use When                   |
|-----------------------------------------|-----------------------------|----------------------------|
| `nn.BCELoss()`                          | Binary Cross Entropy Loss   | Binary classification      |
| `optim.SGD(model.parameters(), lr=...)` | Stochastic Gradient Descent | Simple, standard optimizer |
| `optim.Adam(...)`                       | Adaptive optimizer          | Faster convergence         |
| `loss.backward()`                       | Backpropagation             | After computing loss       |
| `optimizer.step()`                      | Update weights              | After backpropagation      |
| `optimizer.zero_grad()`                 | Reset gradients             | Before backpropagation     |

---

## Evaluation

| Function          | Description                  | Use When               |
|-------------------|------------------------------|------------------------|
| `model.eval()`    | Set model to evaluation mode | Before testing         |
| `model.train()`   | Set model to training mode   | Before training epochs |
| `torch.no_grad()` | Disables gradient tracking   | During testing         |

---

## Regularization and Normalization

| Function                       | Description             | Use When             |
|--------------------------------|-------------------------|----------------------|
| `nn.Dropout(p)`                | Randomly disables units | Reducing overfitting |
| `nn.BatchNorm1d(num_features)` | Normalizes layer input  | Stabilizes training  |

---

## Common Patterns

* **Forward pass**: `outputs = model(inputs)`
* **Loss calculation**: `loss = criterion(outputs, labels)`
* **Backprop & update**:

  ```python
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  ```
* **Binary prediction**: `preds = (outputs > 0.5).float()`

---

Use this as a quick reference while building and experimenting with neural networks in PyTorch. 
