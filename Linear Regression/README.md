# Linear Regression From Scratch

A from-scratch implementation of **Linear Regression** using Python, NumPy, and Gradient Descent.

The goal of this project is to understand the mathematical foundations behind Linear Regression rather than relying directly on a machine-learning library.

## 📌 Overview

Linear Regression is a supervised learning algorithm used to predict a **continuous numerical value**.

The model learns a relationship between an input feature \(X\) and a target variable \(Y\).

For a single feature, the hypothesis is:

$$
\hat{Y} = mX + b
$$

where:

* \(m\) = weight/slope
* \(b\) = bias/intercept
* \(\hat{Y}\) = predicted value

---

## 🧠 Concepts Implemented

* Data loading and preprocessing
* Feature normalization
* Linear Regression
* Mean Squared Error-based cost function
* Partial derivatives
* Gradient Descent
* Weight and bias updates
* Prediction
* Denormalization of predictions
* Comparison with Scikit-learn

---

## 📐 Cost Function

The cost function used during training is:

$$
J(m,b)=\frac{1}{2n}\sum_{i=1}^{n}(\hat{y}_i-y_i)^2
$$

The objective of Gradient Descent is to minimize this cost.

---

### Gradients

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


## 🔄 Training Process

```text
Input Data
    ↓
Normalize Features
    ↓
Initialize m and b
    ↓
Calculate Prediction
    ↓
Calculate Loss
    ↓
Calculate Gradients
    ↓
Update m and b
    ↓
Repeat
    ↓
Final Model
```

---

## 🛠️ Technologies

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---

## 📁 Project Structure

```text
Linear-Regression/
│
├── linear_regression.ipynb
└── README.md
```

---

## 🔬 Validation

The implementation was compared with **Scikit-learn's LinearRegression** model.

The learned parameters from the from-scratch implementation closely match the parameters obtained from Scikit-learn, validating the Gradient Descent implementation.

---

## 📈 Visualization

The notebook includes visualization of the training process and model predictions.

The loss should decrease as Gradient Descent updates the model parameters and approaches the minimum of the cost function.

---

## 🎯 Learning Objective

This project was built to understand how Linear Regression works **under the hood**, including:

1. How predictions are generated.
2. How the loss is calculated.
3. How derivatives provide the direction of optimization.
4. How Gradient Descent updates model parameters.
5. How the learned model can be validated against a standard machine-learning implementation.

This implementation is part of my **Machine Learning From Scratch** learning journey.
