# Improving Your Neural Network

Now that you’ve trained and tested a basic neural network, it’s time to improve it. This section introduces common
techniques to boost model performance, stability, and accuracy.

---

## What We'll Cover

* Adjusting network architecture
* Using better optimizers
* Tuning hyperparameters
* Applying regularization
* Adding batch normalization or dropout

---

## 1. Network Architecture Tweaks

Experiment with adding more layers or increasing hidden layer sizes:

```python
class ImprovedNet(nn.Module):
    def __init__(self):
        super(ImprovedNet, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(10, 5)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
```

---

## 2. Try Better Optimizers

`SGD` is a good starting point, but `Adam` is often more efficient:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

---

## 3. Tune Hyperparameters

Key parameters to experiment with:

* Learning rate (e.g., `0.01`, `0.001`, `0.0001`)
* Batch size (e.g., `2`, `4`, `8`)
* Number of epochs
* Hidden layer sizes

Use tools like grid search or random search to test combinations.

---

## 4. Apply Regularization

Regularization can prevent overfitting.

### Dropout Example:

```python
self.dropout = nn.Dropout(p=0.5)
...
x = self.dropout(x)
```

---

## 5. Add Batch Normalization

This can speed up training and stabilize learning:

```python
self.bn1 = nn.BatchNorm1d(10)
...
x = self.bn1(x)
```

Place after linear layers, before activation functions.

---

## Summary

You’ve learned how to:

* Enhance your model’s architecture
* Apply regularization techniques
* Tune hyperparameters and try new optimizers

