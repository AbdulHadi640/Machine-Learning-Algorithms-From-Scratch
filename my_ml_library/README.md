# ML Library Core Models (`my_ml_library.models`)

This package houses foundational machine learning algorithms implemented entirely from scratch using NumPy. Each module is designed to provide clean, modular, and extensible implementations for educational and applied experimentation.

---

## Module Directory Structure

```text
my_ml_library/
└── models/
    ├── __init__.py                      # Package initialization and direct class exposure
    ├── linear.py                        # Standard Linear Regression implementation
    ├── logistic.py                      # Standard Logistic Regression implementation
    ├── PolynomialRegression.py          # Polynomial feature expansion and modeling
    ├── regularized_logistic_Regression.py # L1/L2 Regularized Logistic Regression variants
    └── regularized_Regression.py        # Regularized Linear Regression (Ridge/Lasso)

```

---

## Module Reference

| Module File | Core Class / Purpose | Key Features |
| --- | --- | --- |
| **`linear.py`** | `LinearRegression` | Ordinary Least Squares via gradient descent, cost tracking, and weight optimization. |
| **`logistic.py`** | `LogisticRegression` | Binary classification using standard sigmoid activation and binary cross-entropy loss. |
| **`PolynomialRegression.py`** | `PolynomialRegression` | Automated multi-degree polynomial feature transformation and scaling utilities. |
| **`regularized_logistic_Regression.py`** | `L1LogisticRegression`, `L2LogisticRegression` | Regularized logistic classifiers incorporating absolute (Lasso) and squared (Ridge) weight penalties. |
| **`regularized_Regression.py`** | `RidgeRegression`, `LassoRegression` | Regularized linear regression models preventing overfitting via coefficient shrinkage. |

---

## Usage Example

Import desired models directly from the package namespace once `__init__.py` exposes them:

```python
from my_ml_library.models import L2LogisticRegression, PolynomialRegression

# 1. Generate polynomial features
poly = PolynomialRegression(degree=3)
X_poly = poly.PolynomialFeatures(X_train)

# 2. Train regularized logistic regression model
model = L2LogisticRegression(learning_rate=0.01, epochs=1000, lamda=1.0)
model.fit(X_poly, Y_train)

# 3. Generate predictions
y_pred = model.predict(X_test_poly)

```