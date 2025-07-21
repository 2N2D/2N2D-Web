# Environment Setup for Building Neural Networks (with PyTorch)

Before we can start building and training neural networks, we need to set up our development environment. This guide
walks you through the tools and libraries needed to get started.

---

## Prerequisites

* Basic familiarity with Python programming
* A computer with internet access
* Some curiosity and enthusiasm to learn!

---

## 1. Install Python

We recommend using Python 3.8 or later.

### Recommended: Install via Anaconda (Beginner-Friendly)

Anaconda comes with Python, Jupyter, and common data science packages.

1. Go to [https://www.anaconda.com/download](https://www.anaconda.com/download)
2. Download the version for your operating system.
3. Install it following the instructions.
4. Open the **Anaconda Navigator** or **Anaconda Prompt**.

### Alternative: Install Python Manually

1. Download from [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Make sure to **check the box**: "Add Python to PATH" during installation.
3. Open your terminal or command prompt and run:

```bash
python --version
```

---

## 2. Choose an IDE or Editor

Pick a place to write and run your code:

* **Jupyter Notebook** – Great for experiments and tutorials
* **VS Code** – Versatile and powerful (install Python extension)
* **PyCharm** – Feature-rich IDE for professional projects

---

## 3. Install Required Libraries

We'll use `PyTorch` for building neural networks.

### Using pip (in terminal or Anaconda Prompt):

First, find your system's compatible installation command
at: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Or install with CPU support only:

```bash
pip install torch torchvision torchaudio
```

Also install common utilities:

```bash
pip install numpy matplotlib
```

If you're using a Jupyter Notebook:

```python
!pip
install
torch
torchvision
torchaudio
numpy
matplotlib
```

> If you encounter issues, consider using a virtual environment or Conda environment.

---

## 4. Set Up a Project Folder (Optional)

It's good practice to keep your code organized:

```bash
neural-networks-course/
├── 01_intro.md
├── 02_how_they_work.md
├── 03_setup.md
├── notebooks/
│   └── basic_network.ipynb
├── data/
└── models/
```

---

## 5. Test Your Installation

Run the following to check PyTorch is installed correctly:

```python
import torch

print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```

If no error occurs and you see a version number (e.g. `2.1.0`) and CUDA shows as available (if you have a GPU), you're
all set!

---

## Next Step

Now that your environment is ready, let's move on to building your **first neural network** in the next section
