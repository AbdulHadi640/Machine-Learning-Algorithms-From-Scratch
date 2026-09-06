# Custom Ridge (L2) Regression From Scratch

A clean, object-oriented implementation of **Ridge Regression (L2 Regularization)** built entirely from scratch using Python and NumPy, applied to the Real Estate Valuation dataset. This repository demonstrates core machine learning principles, including vectorization, analytical gradient descent optimization with L2 shrinkage penalties, and polynomial feature expansion without relying on high-level machine learning libraries like scikit-learn.

---

## Project Structure

```text
├── Datasets/
│   └── Real_estate_valuation.xlsx    # Real Estate Valuation dataset
├── RidgeRegression_model.py            # Object-Oriented RidgeRegression class module
├── RidgeRegression.ipynb               # Step-by-step Ridge implementation, pipeline & evaluation notebook
└── README.md                           # Project documentation

```

---

## Mathematical Formulations

### 1. Cost Function (L2 Regularization)

Ridge regression adds an L2 penalty term to the Mean Squared Error (MSE) objective function to shrink coefficient magnitudes and mitigate multicollinearity:

$$J(W, b) = \frac{1}{2n} \sum (\hat{Y} - Y)^2 + \frac{\lambda}{2n} \sum W^2$$

### 2. Gradient Descent Updates

The analytical partial derivatives of the cost function with respect to the weights ($W$) and bias ($b$) are defined as:

$$\frac{\partial J}{\partial W} = \frac{1}{n} X^T (\hat{Y} - Y) + \frac{\lambda}{n} W$$

$$\frac{\partial J}{\partial b} = \frac{1}{n} \sum (\hat{Y} - Y)$$

---

## Getting Started

### Prerequisites

Ensure you have the following packages installed in your Python environment:

```bash
pip install numpy pandas matplotlib openpyxl

```

### Usage Example

You can import and use the object-oriented `RidgeRegression` class directly in your pipeline:

```python
import numpy as np
from RidgeRegression_model import RidgeRegression

# Initialize and train the Ridge model
ridge_model = RidgeRegression(lamda=3, learning_rate=0.01, epochs=500)
ridge_model.fit(X_scaled, Y_scaled)

# Make predictions
predictions = ridge_model.predict(X_scaled)

```

---

## Key Features & Highlights

* **From-Scratch Vectorization**: Optimized matrix operations via NumPy for efficient dot products and residual computations.
* **L2 Shrinkage Penalty**: Directly integrates regularization parameters into gradient updates to prevent overfitting.


* **Pipeline Integration**: Seamlessly pairs with polynomial feature expansion features to model complex, non-linear relationships.
* **Diagnostic Visualization**: Employs dual-panel Matplotlib figures to monitor training loss convergence curves and evaluate actual versus predicted target alignments against an ideal 1:1 fit line.