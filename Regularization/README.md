# Machine-Learning-Algorithms-From-Scratch: Regularization Module

A comprehensive, production-style module implementing **Ridge (L2)** and **Lasso (L1)** regression models entirely from scratch using Python and NumPy. This repository avoids high-level machine learning abstractions like `scikit-learn` to showcase the fundamental mathematical, vectorization, and optimization principles driving regularized linear models.

---

## Repository Architecture

Based on your directory structure, the repository is organized into distinct subdirectories for datasets, model classes, and experimentation workflows:

```text
Regularization/
├── Datasets/
│   └── Real_estate_valuation.xlsx    # Real Estate Valuation dataset
├── Lasso Regression/
│   ├── LassoRegression.py            # Object-Oriented LassoRegression class module
│   ├── LassoRegression.ipynb         # Step-by-step training, pipeline & evaluation notebook
│   └── README.md                     # Lasso-specific documentation
└── Ridge Regression/
    ├── RidgeRegression_model.py      # Object-Oriented RidgeRegression class module
    ├── RidgeRegression.ipynb         # Step-by-step training, pipeline & evaluation notebook
    └── README.md                     # Ridge-specific documentation

```

---

## Comprehensive Guide to Regularization

In machine learning, standard linear regression attempts to minimize the Mean Squared Error (MSE) between actual targets ($Y$) and predicted targets ($\hat{Y}$). However, when models encounter high-dimensional data, noisy features, or multicollinearity, unconstrained models tend to assign excessively large weights to coefficients. This leads to **overfitting**—where the model memorizes the training data noise instead of learning generalizable patterns.

**Regularization** solves this by adding a penalty term to the loss function, discouraging overly complex models by penalizing large coefficient magnitudes.

---

### 1. Ridge Regression (L2 Regularization)

Ridge regression modifies the standard MSE cost function by adding a penalty proportional to the **squared magnitude** of the weight coefficients (the L2 norm).

#### Mathematical Formulation

The regularized objective (cost) function for Ridge regression is:

$$J(W, b) = \frac{1}{2n} \sum_{i=1}^{n} (\hat{Y}^{(i)} - Y^{(i)})^2 + \frac{\lambda}{2n} \sum_{j=1}^{d} W_j^2$$

Where:

* $n$ is the number of samples.
* $\hat{Y} = XW + b$ is the linear prediction hypothesis.
* $\lambda$ (Lambda) is the hyperparameter controlling the strength of regularization.
* $\sum W^2$ is the squared L2 penalty norm.

#### Gradient Descent Optimization

To minimize this cost function via gradient descent, we take the analytical partial derivatives with respect to weights ($W$) and bias ($b$):

$$\frac{\partial J}{\partial W} = \frac{1}{n} X^T (\hat{Y} - Y) + \frac{\lambda}{n} W$$

$$\frac{\partial J}{\partial b} = \frac{1}{n} \sum (\hat{Y} - Y)$$

#### Key Characteristics of Ridge

* **Coefficient Shrinkage:** The L2 penalty shrinks coefficients continuously toward zero, but never makes them strictly absolute zero.
* **Multicollinearity Management:** It distributes weight evenly among correlated features, stabilizing variance and preventing catastrophic weight inflation.

---

### 2. Lasso Regression (L1 Regularization)

Lasso (Least Absolute Shrinkage and Selection Operator) regression introduces a penalty proportional to the **absolute magnitude** of the weight coefficients (the L1 norm).

#### Mathematical Formulation

The regularized objective function for Lasso regression is:

$$J(W, b) = \frac{1}{2n} \sum_{i=1}^{n} (\hat{Y}^{(i)} - Y^{(i)})^2 + \frac{\lambda}{n} \sum_{j=1}^{d} \vert{}W_j\vert{}$$

#### Gradient Descent Optimization & The L1 Subgradient

Because the absolute value function $\vert{}W\vert{}$ is non-differentiable at zero ($W = 0$), Lasso utilizes a subgradient defined by the sign function:

$$\frac{\partial J}{\partial W} = \frac{1}{n} X^T (\hat{Y} - Y) + \frac{\lambda}{n} \text{sign}(W)$$

$$\frac{\partial J}{\partial b} = \frac{1}{n} \sum (\hat{Y} - Y)$$

Where `np.sign(W)` returns $+1$ for positive weights, $-1$ for negative weights, and handles directional shrinkage steps.

#### Key Characteristics of Lasso

* **Automatic Feature Selection (Sparsity):** Unlike L2, the constant step size of the L1 derivative forces coefficients of irrelevant or redundant features to collapse entirely to **zero**.
* **Model Interpretability:** By zeroing out uninformative weights, Lasso acts as a built-in feature selector, leaving behind a sparse model highlighting only the most impactful drivers.

---

## Implementation Highlights

* **From-Scratch Vectorization:** Matrix operations are optimized using NumPy to compute efficient dot products, residuals, and cost tracking.
* **Z-Score Normalization:** Features and targets are scaled to ensure smooth, stable gradient descent convergence without scale dominance.
* **Modular Code Architecture:** Logic is cleanly separated into reusable class modules (`LassoRegression.py`, `RidgeRegression_model.py`) and exploratory Jupyter notebooks.

---

## Getting Started & Usage

### Prerequisites

Install the required packages to run the notebooks and scripts:

```bash
pip install numpy pandas matplotlib openpyxl

```

### Python Implementation Example

```python
import numpy as np
import pandas as pd

# Load dataset
df = pd.read_excel("Datasets/Real_estate_valuation.xlsx")
X = df.drop("Y house price of unit area", axis=1).to_numpy()
Y = df["Y house price of unit area"].to_numpy().reshape(-1, 1)

# Normalization
X_norm = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, keepdims=True)
Y_norm = (Y - np.mean(Y)) / np.std(Y)

# --- Train Ridge Regression ---
from Ridge_Regression.RidgeRegression_model import RidgeRegression
ridge = RidgeRegression(lamda=3, learning_rate=0.01, epochs=500)
ridge.fit(X_norm, Y_norm)

# --- Train Lasso Regression ---
from Lasso_Regression.LassoRegression import LassoRegression
lasso = LassoRegression(lamda=1, learning_rate=0.001, epochs=10000)
lasso.fit(X_norm, Y_norm)

```
