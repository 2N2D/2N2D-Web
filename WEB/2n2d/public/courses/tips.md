# Neural Network Design Tips & Function Guide

This guide gives you a quick reference on when to use key PyTorch functions and how to think about building effective
neural network structures.

---

## When to Use Which PyTorch Function

| Function                | Use When                               | Notes                                            |
|-------------------------|----------------------------------------|--------------------------------------------------|
| `nn.Linear(in, out)`    | To map features between layers         | Defines fully connected layers                   |
| `nn.ReLU()`             | After Linear layers                    | Adds non-linearity, prevents vanishing gradients |
| `nn.Sigmoid()`          | Output layer for binary classification | Produces probabilities between 0 and 1           |
| `nn.BCELoss()`          | Binary classification problems         | Requires output to use sigmoid                   |
| `torch.optim.SGD(...)`  | For controlled, step-based learning    | Requires careful learning rate tuning            |
| `torch.optim.Adam(...)` | For faster, adaptive optimization      | A good default for most tasks                    |
| `model.train()`         | Before training starts                 | Enables dropout, batch norm, etc.                |
| `model.eval()`          | Before evaluating or testing           | Disables dropout, batch norm                     |
| `torch.no_grad()`       | During inference                       | Disables gradient calculation for efficiency     |

---

## Tips for Designing Neural Networks

### 1. Start Simple

* Begin with **1–2 hidden layers**
* Use **4–10 neurons** per layer for small tasks
* Simple architecture = easier to debug

### 2. Activation Functions

* Use **ReLU** in hidden layers (fast, effective)
* Use **Sigmoid** or **Softmax** in the output layer depending on the task:

    * `Sigmoid` → Binary classification
    * `Softmax` → Multi-class classification

### 3. Overfitting Prevention

* Add **Dropout** layers (e.g., `nn.Dropout(p=0.5)`) to randomly deactivate neurons
* Use **Batch Normalization** (`nn.BatchNorm1d`) to stabilize learning

### 4. Hyperparameters to Tune

* **Learning rate** (`lr`) – Start with `0.001` or `0.01`
* **Batch size** – Try `8`, `16`, `32`, depending on dataset size
* **Epochs** – Watch training/validation loss to stop at the right time
* **Hidden layer size** – Try increasing if underfitting, reducing if overfitting

### 5. Evaluate Smart

* Always **split data** into training and testing sets
* Track **accuracy**, **precision**, or **loss curves** depending on your goal

---

## Quick Checklist

* [x] Use `model.train()` before training
* [x] Use `model.eval()` and `torch.no_grad()` during evaluation
* [x] Normalize your inputs (e.g., with `StandardScaler`)
* [x] Experiment in small steps: one change at a time

---

Use this guide alongside your project to make smart design decisions and accelerate learning! 🚀
