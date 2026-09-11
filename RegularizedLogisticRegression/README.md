# Regularized Logistic Regression & Polynomial Feature Engineering From Scratch

A comprehensive machine learning module implementing **L1 (Lasso)** and **L2 (Ridge)** Regularized Logistic Regression algorithms completely from scratch using **NumPy**, accompanied by a non-linear **Polynomial Feature Generator**.

---

## Directory Architecture

```text
E:\ML Algorithms From Scratch\
│
├── my_ml_library\
│   └── models.py                  # Core PolynomialRegression feature expansion engine
├── RegularizedLogisticRegression\
│   ├── L1LogRegression.py         # L1 Lasso regularized binary classifier
│   ├── L2LogRegression.py         # L2 Ridge regularized binary classifier
│   └── regularization.ipynb       # Interactive experimentation and workflow notebook
└── README.md

```

---

## Mathematical Foundations & Implementation Details

### 1. Polynomial Feature Expansion (`PolynomialRegression`)

To allow linear decision boundaries to capture non-linear patterns, input features $\mathbf{X}$ are expanded up to a specified polynomial degree $d$:


$$\mathbf{X}_{\text{poly}} = \left[ x_1, x_1^2, \dots, x_1^d, \; x_2, x_2^2, \dots, x_2^d, \; \dots \right]$$

### 2. Numerically Stable Sigmoid Activation

To prevent floating-point overflow during exponential evaluations for large negative or positive inputs, the hypothesis uses a piece-wise stabilized formulation:


$$\sigma(z) = \begin{cases} \frac{1}{1 + e^{-z}} & \text{if } z \ge 0 \\ \frac{e^z}{1 + e^z} & \text{if } z < 0 \end{cases}$$

### 3. L1 Regularization (Lasso)

* **Objective Function:** Adds an absolute penalty on weights to encourage sparsity (driving irrelevant feature weights to zero):

$$J(\mathbf{W}, b) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right] + \frac{\lambda}{n} \sum_{j} \vert{}w_j\vert{}$$


* **Gradient Update:**

$$\frac{\partial J}{\partial w_j} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)} + \frac{\lambda}{n} \text{sign}(w_j)$$



### 4. L2 Regularization (Ridge)

* **Objective Function:** Adds a squared penalty on weights to smoothly shrink coefficients and prevent overfitting:

$$J(\mathbf{W}, b) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right] + \frac{\lambda}{2n} \sum_{j} w_j^2$$


* **Gradient Update:**

$$\frac{\partial J}{\partial w_j} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)} + \frac{\lambda}{n} w_j$$



---

## Notebook Workflow Guide (`regularization.ipynb`)

Inside your interactive Jupyter notebook (`regularization.ipynb`), structure your execution across cells following this pipeline:

1. **Environment Setup & Path Injection:** Ensure the project root directory is appended to `sys.path` so Python can locate your custom modules directly without import errors.
2. **Data Transformation:** Import `PolynomialRegression` from your library, define your polynomial degree (e.g., `degree=3`), and transform your training and testing features.
3. **Feature Standardization:** Compute the mean and standard deviation strictly from the training polynomial features, then normalize both train and test sets (`(X_poly - mean) / (std + 1e-8)`) to maintain numerical stability during gradient descent.
4. **Model Initialization & Training:** Instantiate either `L1LogisticRegression` or `L2LogisticRegression` with tuned hyperparameters (`learning_rate`, `epochs`, `lamda`) and call the `.fit()` method on your scaled matrices.
5. **Evaluation:** Use the `.predict()` method to generate probability outputs for your test sets.

---

## Hyperparameter Reference Guide

| Hyperparameter | Description | Recommended Starting Value |
| --- | --- | --- |
| `learning_rate` ($\alpha$) | Controls step size during gradient descent updates. | `0.01` to `0.05` |
| `epochs` | Total number of full training iterations over the dataset. | `1000` |
| `lamda` ($\lambda$) | Regularization strength penalty controlling model complexity. | `1.0` to `3.0` |
| `degree` | Power degree for polynomial feature engineering expansion. | `2` to `3` |