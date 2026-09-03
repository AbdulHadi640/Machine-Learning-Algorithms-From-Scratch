# Multiple Linear Regression From Scratch

A from-scratch implementation of **Multiple Linear Regression** using Python and NumPy.

This project extends the concepts learned from Single Linear Regression to multiple input features and focuses on **vectorized matrix operations, Gradient Descent, normalization, and parameter updates**.

## Overview

Multiple Linear Regression predicts a continuous target using multiple input features.

The general model is:

$$
\hat{Y}
=
w_1x_1+w_2x_2+\cdots+w_dx_d+b
$$

Using matrix notation:

$$
\hat{Y}=WX+b
$$

where:

* $W$ = weight matrix
* $X$ = input feature matrix
* $b$ = bias
* $\hat{Y}$ = predicted target
* $d$ = number of features
* $n$ = number of samples

## Matrix Orientation

Two equivalent matrix orientations were explored during implementation.

### Approach 1 — Features × Samples

For this implementation:

* Rows represent features
* Columns represent samples

Therefore:

$$
X\in\mathbb{R}^{d\times n}
$$

$$
W\in\mathbb{R}^{1\times d}
$$

For the current dataset:

$$
X=(6\times414)
$$

$$
W=(1\times6)
$$

### Prediction

The prediction is:

$$
\hat{Y}=WX+b
$$

Shape calculation:

$$
(1\times6)(6\times414)
=
(1\times414)
$$

Therefore, one prediction is produced for each sample.

### Gradient with Respect to Weights

For this orientation:

$$
\frac{\partial J}{\partial W}
=
\frac{1}{n}
(\hat{Y}-Y)X^T
$$

Shape:

$$
(1\times414)(414\times6)
=
(1\times6)
$$

Thus, one gradient is produced for every weight.

In NumPy:

```python
dw = ((Y_hat - Y) @ X.T) / n
```

### Gradient with Respect to Bias

$$
\frac{\partial J}{\partial b}
=
\frac{1}{n}
\sum_{i=1}^{n}
(\hat{y}_i-y_i)
$$

## Approach 2 — Samples × Features

The second approach follows the conventional machine-learning orientation:

* Rows represent samples
* Columns represent features

Therefore:

$$
X\in\mathbb{R}^{n\times d}
$$

$$
W\in\mathbb{R}^{d\times1}
$$

For the same dataset:

$$
X=(414\times6)
$$

$$
W=(6\times1)
$$

### Prediction

The prediction is:

$$
\hat{Y}=XW+b
$$

Shape:

$$
(414\times6)(6\times1)
=
(414\times1)
$$

### Gradient with Respect to Weights

For this orientation:

$$
\frac{\partial J}{\partial W}
=
\frac{1}{n}
X^T(\hat{Y}-Y)
$$

Shape:

$$
(6\times414)(414\times1)
=
(6\times1)
$$

In NumPy:

```python
dw = (X.T @ (Y_hat - Y)) / n
```

### Gradient with Respect to Bias

$$
\frac{\partial J}{\partial b}
=
\frac{1}{n}
\sum_{i=1}^{n}
(\hat{y}_i-y_i)
$$

## Cost Function

The cost function used for training is:

$$
J(W,b)
=
\frac{1}{2n}
\sum_{i=1}^{n}
(\hat{y}_i-y_i)^2
$$

The objective is:

$$
\boxed{\min_{W,b}J(W,b)}
$$

## Gradient Descent

The weights and bias are updated repeatedly:

$$
W:=W-\alpha\frac{\partial J}{\partial W}
$$

$$
b:=b-\alpha\frac{\partial J}{\partial b}
$$

where $\alpha$ is the learning rate.

## Feature Normalization

Each feature is normalized independently.

For the Features × Samples orientation:

$$
X_{norm}
=
\frac{X-\mu_X}{\sigma_X}
$$

where the mean and standard deviation are calculated across the samples of each feature.

Using NumPy:

```python
X_mean = np.mean(X, axis=1, keepdims=True)
X_std = np.std(X, axis=1, keepdims=True)

X_norm = (X - X_mean) / X_std
```

For the Samples × Features orientation, the same operation is performed across rows using the feature axis.

## Training Process

```text
Input Dataset
      ↓
Select Features and Target
      ↓
Train/Test Split
      ↓
Normalize Features
      ↓
Initialize W and b
      ↓
Calculate Y_hat
      ↓
Calculate Cost
      ↓
Calculate dW and db
      ↓
Update W and b
      ↓
Repeat
      ↓
Denormalize Predictions
      ↓
Evaluate Model
```

## Vectorization

Instead of calculating every weight gradient separately using individual loops, matrix multiplication calculates all weight gradients simultaneously.

For example:

$$
dW=\frac{1}{n}(\hat{Y}-Y)X^T
$$

produces the gradient for all weights in one vectorized operation.

This makes the implementation cleaner and more efficient.

## Validation

The from-scratch implementation was compared with Scikit-learn's `LinearRegression`.

The learned weights from the two approaches were very close to the Scikit-learn coefficients, validating the implementation.

## Key Learning

The most important concept in this implementation was understanding that the mathematical operation remains the same even when the matrix orientation changes.

For example:

```text
Features × Samples:

W @ X

(1 × d) @ (d × n)
→ (1 × n)
```

while:

```text
Samples × Features:

X @ W

(n × d) @ (d × 1)
→ (n × 1)
```

Both produce one prediction for every sample.

## Technologies

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

## Project Structure

```text
Multiple Linear Regression/
│
├── multiple_linear_regression.ipynb
└── README.md
```

## Learning Objective

This implementation was built to understand:

1. Multiple Linear Regression mathematically.
2. Matrix-based prediction.
3. Matrix multiplication and shape compatibility.
4. Vectorized gradient calculation.
5. Gradient Descent for multiple parameters.
6. Feature normalization.
7. Denormalization of predictions.
8. Different matrix orientations.
9. Validation against Scikit-learn.
