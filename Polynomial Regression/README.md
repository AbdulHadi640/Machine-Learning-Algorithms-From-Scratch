# Polynomial & Multivariate Regression from Scratch (NumPy)

A pure NumPy implementation of Univariate and Multivariate Polynomial Regression using Batch Gradient Descent, built without Scikit-Learn optimization abstractions.

---

## 📌 Overview

This project extends standard linear regression to capture non-linear dependencies in tabular data. It implements an explicit polynomial feature-expansion engine, robust normalization pipelines, and vectorized gradient updates applied to real estate valuation modeling.

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

## 🚀 Getting Started

### Prerequisites
* Python 3.9+
* `numpy`, `pandas`, `matplotlib`, `openpyxl`

### Installation
```bash
git clone [https://github.com/your-username/polynomial-regression-from-scratch.git](https://github.com/your-username/polynomial-regression-from-scratch.git)
cd polynomial-regression-from-scratch
pip install numpy pandas matplotlib openpyxl