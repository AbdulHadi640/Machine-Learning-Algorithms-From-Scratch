# Machine Learning Regularization From Scratch: Ridge & Lasso

A comprehensive, object-oriented module implementing **Ridge (L2)** and **Lasso (L1)** regression models built entirely from scratch using Python and NumPy, applied to the Real Estate Valuation dataset. This repository avoids high-level machine learning abstractions like `scikit-learn` to demonstrate the foundational mathematical mechanics, vectorization, gradient descent optimization, and polynomial feature expansion driving regularized linear models.

---

## Directory Structure

```text
Regularization/
├── Datasets/
│   └── Real_estate_valuation.xlsx    # Real Estate Valuation dataset[cite: 3, 5]
├── Lasso Regression/
│   ├── LassoRegression.py            # Object-Oriented LassoRegression class module[cite: 3]
│   ├── LassoRegression.ipynb         # Step-by-step Lasso implementation & analysis notebook[cite: 3]
│   └── README.md                     # Lasso-specific documentation
└── Ridge Regression/
    ├── RidgeRegression_model.py      # Object-Oriented RidgeRegression class module[cite: 5]
    ├── RidgeRegression.ipynb         # Step-by-step Ridge implementation & evaluation notebook[cite: 5]
    └── README.md                     # Ridge-specific documentation

```

---

## The Theory of Regularization

Standard linear regression minimizes the Mean Squared Error (MSE) to fit a hyperplane to data. However, when models encounter high-dimensional spaces, noisy inputs, or multicollinearity, unconstrained models often learn excessively large weight coefficients. This leads to **overfitting**, where the model memorizes training noise rather than capturing generalizable trends.

**Regularization** prevents this by adding a penalty term to the loss function, penalizing large coefficient magnitudes to constrain model complexity and reduce variance.

---

## Mathematical Formulations & Core Mechanisms

### 1. Ridge Regression (L2 Regularization)

Ridge regression introduces a penalty proportional to the **squared magnitude** of the weight coefficients (the L2 norm).

* **Cost Function:**

$$J(W, b) = \frac{1}{2n} \sum (\hat{Y} - Y)^2 + \frac{\lambda}{2n} \sum W^2$$



* **Gradient Updates:**

$$\frac{\partial J}{\partial W} = \frac{1}{n} X^T (\hat{Y} - Y) + \frac{\lambda}{n} W$$



$$\frac{\partial J}{\partial b} = \frac{1}{n} \sum (\hat{Y} - Y)$$



* **Core Benefit:** Shrinks coefficients smoothly toward zero and handles multicollinearity by distributing weights evenly across correlated features.



---

### 2. Lasso Regression (L1 Regularization)

Lasso regression introduces a penalty proportional to the **absolute magnitude** of the weight coefficients (the L1 norm).

* **Cost Function:**

$$J(W, b) = \frac{1}{2n} \sum (\hat{Y} - Y)^2 + \frac{\lambda}{n} \sum \vert{}W\vert{}$$



* **Gradient Updates (with L1 Subgradient):**

$$\frac{\partial J}{\partial W} = \frac{1}{n} X^T (\hat{Y} - Y) + \frac{\lambda}{n} \text{sign}(W)$$



$$\frac{\partial J}{\partial b} = \frac{1}{n} \sum (\hat{Y} - Y)$$


* **Core Benefit:** Because the L1 derivative is a constant step size determined by `np.sign(W)`, Lasso can drive coefficients of uninformative features entirely to **zero**, performing explicit feature selection.



---

## Getting Started

### Prerequisites

Ensure you have the required packages installed in your Python environment:

```bash
pip install numpy pandas matplotlib openpyxl

```

### Usage Example

You can import and use either object-oriented model class directly in your pipeline:

```python
import numpy as np
from RidgeRegression_model import RidgeRegression
from LassoRegression import LassoRegression

# Initialize and train the Ridge model
ridge_model = RidgeRegression(lamda=3, learning_rate=0.01, epochs=500)
ridge_model.fit(X_scaled, Y_scaled)
ridge_preds = ridge_model.predict(X_scaled)

# Initialize and train the Lasso model
lasso_model = LassoRegression(lamda=1, learning_rate=0.01, epochs=10000)
lasso_model.fit(X_scaled, Y_scaled)
lasso_preds = lasso_model.LinearModel(X_scaled)

```

---

## Key Features & Highlights

* **Modular OOP Architecture:** Clean class-based designs (`RidgeRegression_model.py` and `LassoRegression.py`) encapsulating initialization, training loops, and predictions.


* **Custom Feature Engineering:** Integrated support for a custom-built polynomial transformation pipeline to capture non-linear real estate trends.


* **Rigorous Evaluation:** Automated workflows featuring Z-score normalization, loss convergence tracking over epochs, and 1:1 actual vs. predicted distribution visualizations.
