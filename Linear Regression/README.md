# Linear Regression From Scratch

A from-scratch implementation of **Linear Regression** using Python, NumPy, Pandas, Matplotlib, and Gradient Descent.

The purpose of this implementation is to understand the mathematical and computational foundations of Linear Regression instead of directly relying on a machine-learning library.

## Overview

Linear Regression is a supervised learning algorithm used to predict a **continuous numerical target**.

For a single feature, the model is:

$$
\hat{Y} = mX + b
$$

where:

* $m$ = weight / slope
* $b$ = bias / intercept
* $\hat{Y}$ = predicted value

## Cost Function

The Mean Squared Error-based cost function used in the implementation is:

$$
J(m,b)
=
\frac{1}{2n}
\sum_{i=1}^{n}
(\hat{y}_i-y_i)^2
$$

The objective is to minimize:

$$
\boxed{\min_{m,b} J(m,b)}
$$

## Gradients

The gradient with respect to the weight is:

$$
\frac{\partial J}{\partial m}
=
\frac{1}{n}
\sum_{i=1}^{n}
(\hat{y}_i-y_i)x_i
$$

The gradient with respect to the bias is:

$$
\frac{\partial J}{\partial b}
=
\frac{1}{n}
\sum_{i=1}^{n}
(\hat{y}_i-y_i)
$$

## Gradient Descent

The parameters are updated iteratively using:

$$
m := m-\alpha\frac{\partial J}{\partial m}
$$

$$
b := b-\alpha\frac{\partial J}{\partial b}
$$

where $\alpha$ is the learning rate.

### Learning Rate

The learning rate controls the size of each update.

* Very small learning rate → slow convergence
* Very large learning rate → may overshoot or diverge
* Appropriate learning rate → gradual convergence toward the minimum

## Data Normalization

The feature and target were normalized before training:

$$
X_{norm}
=
\frac{X-\mu_X}{\sigma_X}
$$

$$
Y_{norm}
=
\frac{Y-\mu_Y}{\sigma_Y}
$$

Normalization helps Gradient Descent converge more effectively.

## Denormalization

After obtaining predictions in normalized units, predictions are converted back to the original target scale:

$$
Y_{original}
=
Y_{norm}\sigma_Y+\mu_Y
$$

This allows the final predictions to be interpreted in the original units of the dataset.

## Training Process

```text
Input Data
    ↓
Normalize X and Y
    ↓
Initialize m and b
    ↓
Calculate Prediction
    ↓
Calculate Cost
    ↓
Calculate Gradients
    ↓
Update m and b
    ↓
Repeat for multiple iterations
    ↓
Denormalize Predictions
    ↓
Evaluate Model
```

## Implementation

The implementation was built using:

* Python
* NumPy
* Pandas
* Matplotlib

Scikit-learn was used only to compare the learned parameters with a standard implementation.

## Validation

The parameters learned using the from-scratch implementation were compared with Scikit-learn's `LinearRegression`.

The learned parameters were very close to the Scikit-learn results, providing a practical validation of the Gradient Descent implementation.

## Project Structure

```text
Linear Regression/
│
├── linear_regression.ipynb
└── README.md
```

## Learning Objective

This implementation focuses on understanding Linear Regression from the ground up:

1. How predictions are calculated.
2. How the cost function measures prediction error.
3. How derivatives provide the direction of optimization.
4. How Gradient Descent updates model parameters.
5. Why normalization is useful.
6. How normalized predictions are converted back to original units.
7. How a from-scratch implementation can be validated against Scikit-learn.
