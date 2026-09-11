# Polynomial & Multivariate Regression from Scratch (NumPy)

A pure NumPy implementation of Univariate and Multivariate Polynomial Regression using Batch Gradient Descent, built without Scikit-Learn optimization abstractions.

---

## 📌 Overview

This project extends standard linear regression to capture non-linear dependencies in tabular data. It implements an explicit polynomial feature-expansion engine, robust normalization pipelines, and vectorized gradient updates applied to predictive modeling.

---

## 📐 Mathematical Formulation

### 1. Polynomial Feature Mapping

For an input feature vector $\mathbf{x} = [x_1, x_2, \dots, x_d]^T$ expanded to degree $k$:


$$\Phi(\mathbf{x}) = [x_1, x_1^2, \dots, x_1^k, \dots, x_d, x_d^2, \dots, x_d^k]^T$$

The hypothesis function operates over the expanded feature space:


$$\hat{Y} = \Phi(X)W + b$$

### 2. Cost Function (Mean Squared Error)

$$J(W, b) = \frac{1}{2n} \sum_{i=1}^{n} (\hat{y}^{(i)} - y^{(i)})^2$$

### 3. Vectorized Gradient Descent

$$\frac{\partial J}{\partial W} = \frac{1}{n} \Phi(X)^T (\hat{Y} - Y)$$

$$\frac{\partial J}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}^{(i)} - y^{(i)})$$

Parameters are updated simultaneously:


$$W := W - \alpha \frac{\partial J}{\partial W}, \quad b := b - \alpha \frac{\partial J}{\partial b}$$

---

## ⚙️ Engineering Challenges Solved

* **Numerical Instability & Gradient Explosion:** Exponentiating raw values ($X^2, X^3$) rapidly scales feature magnitudes, leading to `NaN` overflow during matrix dot products. Standardizing the matrix along `axis=0` after expansion with a variance stabilizer ($\epsilon = 10^{-8}$) ensured consistent convergence.
* **Memory-Efficient Concatenation:** Replaced slow `np.append` loops with list accumulation and `np.column_stack` to prevent repeated memory reallocations.
* **Multi-Feature Non-Linear Visualization:** Projected higher-dimensional polynomial fits into interpretable 2D spaces by evaluating isolated feature response curves while holding remaining covariates at their empirical mean.

---

## 📂 Project Architecture

```text
E:\ML Algorithms From Scratch\
│
├── my_ml_library/
│   └── models/
│       ├── __init__.py                      # Exposes package models directly for clean imports
│       ├── linear.py                        # Standard Linear Regression implementation
│       ├── logistic.py                      # Standard Logistic Regression implementation
│       ├── PolynomialRegression.py          # Polynomial feature expansion and modeling engine
│       ├── regularized_logistic_Regression.py # L1/L2 Regularized Logistic Regression variants
│       └── regularized_Regression.py        # Regularized Linear Regression (Ridge/Lasso)
│
├── Linear Regression/
│   ├── Linear Regression.ipynb              # Step-by-step interactive workflow notebook
│   └── LinearRegression.py                  # Standalone implementation script
│
├── PolynomialRegression/
│   ├── Polynomial Regression.ipynb          # Interactive step-by-step workflow notebook
│   ├── PolynomialRegression.py              # Standalone implementation script
│   └── README.md                            # Local module documentation
│
├── LogisticRegression/
│   ├── Logistic Regression.ipynb            # Step-by-step interactive workflow notebook
│   └── LogisticRegression.py                # Standalone implementation script
│
├── RegularizedLogisticRegression/
│   ├── L1LogRegression.py                   # L1 Regularized Logistic Regression class
│   ├── L2LogRegression.py                   # L2 Regularized Logistic Regression class
│   └── regularization.ipynb                 # Interactive experimentation notebook
│
└── README.md

```

---

## 🚀 Getting Started

### Prerequisites

* Python 3.9+
* `numpy`, `pandas`, `matplotlib`, `openpyxl`

### Installation

```bash
git clone https://github.com/your-username/polynomial-regression-from-scratch.git
cd polynomial-regression-from-scratch
pip install numpy pandas matplotlib openpyxl

```

---

## 💻 Usage Example

Import your custom models directly from the package namespace once configured:

```python
import numpy as np
from my_ml_library.models import PolynomialRegression, L2LogisticRegression

# 1. Initialize polynomial feature transformer
poly = PolynomialRegression(degree=3, learning_rate=0.01, epochs=1000)

# 2. Fit and train on raw training data
poly.fit(X_train, Y_train)

# 3. Predict on unseen test data using internal auto-normalization
y_pred = poly.LinearModel(X_test)

```