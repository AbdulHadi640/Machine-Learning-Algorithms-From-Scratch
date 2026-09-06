# Custom Lasso (L1) Regression From Scratch

A clean, object-oriented implementation of **Lasso Regression (L1 Regularization)** built entirely from scratch using Python and NumPy, applied to the Real Estate Valuation dataset. This repository demonstrates the foundational mechanics of gradient descent optimization, analytical subgradient derivation for L1 penalties, and polynomial feature expansion without relying on high-level machine learning libraries like scikit-learn.

---

## Project Overview

While standard linear regression minimizes Mean Squared Error (MSE), Lasso regression introduces an L1 regularization penalty ($\lambda \sum \vert{}W\vert{}$). This penalizes absolute weight magnitudes, forcing irrelevant feature coefficients to zero and performing explicit feature selection.

This project covers:

* **Procedural and Object-Oriented Implementations**: Transitioning from functional scripts to a reusable `LassoRegression` class architecture.


* **Mathematical Optimization**: Implementing analytical partial derivatives with the L1 subgradient term (`np.sign(W)`) to stabilize gradient updates.


* **Pipeline Integration**: Extending the model with a degree-3 polynomial feature transformation pipeline.


* **Rigorous Evaluation**: Z-score feature normalization, training loss convergence tracking, and 1:1 actual vs. predicted value visualizations.



---

## Repository Structure

```text
├── Datasets/
│   └── Real_estate_valuation.xlsx    # Real Estate Valuation dataset
├── LassoRegression.py                  # Object-Oriented LassoRegression class module
├── LassoRegression.ipynb               # Jupyter notebook containing step-by-step implementation & analysis
└── README.md                           # Project documentation

```

---

## Mathematical Formulations

### 1. Cost Function (L1 Regularization)

The regularized objective function combines Mean Squared Error with the L1 penalty:

$$J(W, b) = \frac{1}{2n} \sum (\hat{Y} - Y)^2 + \frac{\lambda}{n} \sum \vert{}W\vert{}$$

### 2. Gradient Descent Updates

The partial derivatives of the cost function with respect to the weights ($W$) and bias ($b$) include the L1 subgradient derivative term:

$$\frac{\partial J}{\partial W} = \frac{1}{n} X^T (\hat{Y} - Y) + \frac{\lambda}{n} \text{sign}(W)$$

$$\frac{\partial J}{\partial b} = \frac{1}{n} \sum (\hat{Y} - Y)$$

---

## Getting Started

### Prerequisites

Ensure you have the following libraries installed in your Python environment:

```bash
pip install numpy pandas matplotlib openpyxl

```

### Usage Example

You can import and use the object-oriented `LassoRegression` class directly in your scripts:

```python
import numpy as np
from LassoRegression import LassoRegression

# Initialize and train the model
model = LassoRegression(lamda=1, learning_rate=0.01, epochs=10000)
model.fit(X_scaled, Y_scaled)

# Make predictions
predictions = model.LinearModel(X_scaled)

```

---

## Model Evaluation & Performance

Training the model using gradient descent with proper Z-score scaling and L1 subgradient updates yields stable loss convergence and tight clustering along the 1:1 ideal fit line for housing price predictions.