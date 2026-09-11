# Logistic Regression from Scratch (`LogisticRegression`)

This directory contains a standalone implementation of standard binary logistic regression from scratch using NumPy, paired with an interactive Jupyter notebook for step-by-step model building and evaluation.

---

## Directory Architecture

```text
LogisticRegression/
├── LogisticRegression.py          # Core class implementation (Sigmoid, Cost, Gradients, Fit, Predict)
└── Logistic Regression.ipynb      # Interactive step-by-step workflow and testing notebook

```

---

## Core Components (`LogisticRegression.py`)

### 1. Hypothesis & Sigmoid Activation

Maps linear feature combinations into probability scores between 0 and 1:


$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

### 2. Binary Cross-Entropy Cost Function

Computes classification error protected by a numerical clipping epsilon ($\epsilon = 10^{-15}$) to prevent `log(0)` arithmetic errors:


$$J(\mathbf{W}, b) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

### 3. Gradient Descent Optimization

Iteratively updates weights ($\mathbf{W}$) and bias ($b$) via partial derivatives:


$$\frac{\partial J}{\partial \mathbf{W}} = \frac{1}{n} \mathbf{X}^T (\hat{\mathbf{Y}} - \mathbf{Y})$$

$$\frac{\partial J}{\partial b} = \frac{1}{n} \sum (\hat{\mathbf{Y}} - \mathbf{Y})$$

---

## Step-by-Step Jupyter Workflow (`Logistic Regression.ipynb`)

Inside your interactive notebook located in this folder, structure your execution cells sequentially:

1. **Import Dependencies & Model Class:**
```python
import numpy as np
from LogisticRegression import LogisticRegression

```


2. **Generate or Load Sample Data:**
```python
np.random.seed(42)
X_train = np.random.randn(200, 2)
Y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

```


3. **Initialize and Train:**
```python
model = LogisticRegression(learning_rate=0.1, epochs=1000)
model.fit(X_train, Y_train)

```


4. **Predict & Evaluate:**
```python
probabilities = model.predict(X_train)
predictions = (probabilities >= 0.5).astype(int)

```