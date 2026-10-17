# Decision Tree Classifier From Scratch 🌳

A **Decision Tree Classifier implemented from scratch using Python and NumPy**. This project explores how decision trees select features, calculate impurity, determine optimal split points, and recursively build classification trees without relying on Scikit-learn for the core algorithm.

The implementation supports both **Entropy and Gini Impurity**, categorical and numerical features, and configurable stopping conditions. Its performance is evaluated against Scikit-learn's Decision Tree Classifier on three datasets.

## 📌 Project Overview

Decision trees are supervised machine learning algorithms used for classification and regression. They make predictions by recursively splitting data based on feature values until a stopping condition is reached.

In this project, I implemented the core decision tree algorithm from scratch to understand its internal working, including:

- Entropy and Gini impurity calculations.
- Information gain and feature selection.
- Categorical and numerical feature splitting.
- Automatic selection of numerical thresholds.
- Recursive tree construction.
- Prediction using the constructed tree.
- Comparison with Scikit-learn's implementation.

## ✨ Features

- **From-scratch implementation:** Core tree-building and prediction logic implemented using Python and NumPy.
- **Multiple splitting criteria:** Supports Entropy and Gini Impurity.
- **Categorical features:** Creates separate branches for unique categorical values.
- **Numerical features:** Evaluates candidate thresholds and selects the split with the highest impurity reduction.
- **Automatic feature selection:** Chooses the feature that produces the best information gain or impurity reduction.
- **Recursive construction:** Builds a tree by repeatedly splitting child datasets.
- **Stopping conditions:** Supports maximum depth and minimum samples required to split.
- **Majority-class prediction:** Handles cases where further splitting is not possible.
- **Scikit-learn comparison:** Evaluates the custom implementation against a standard library implementation.

## 🧠 How It Works

### 1. Entropy

Entropy measures the uncertainty or impurity of a dataset.

$$
H(Y)=-\sum_{i=1}^{C}p_i\log_2(p_i)
$$

Where:

- \(C\) is the number of classes.
- \(p_i\) is the proportion of samples belonging to class \(i\).

A pure node has zero entropy because all samples belong to the same class.

### 2. Gini Impurity

Gini impurity measures the probability of incorrectly classifying a randomly selected sample if it is labeled according to the class distribution in the node.

$$
G(Y)=1-\sum_{i=1}^{C}p_i^2
$$

A lower Gini impurity indicates a purer node.

### 3. Information Gain

Information gain measures the reduction in impurity after splitting a dataset.

$$
IG(Y,X)=I(Y)-\sum_{j=1}^{K}\frac{|Y_j|}{|Y|}I(Y_j)
$$

Where:

- \(I(Y)\) is the parent node's impurity.
- \(K\) is the number of child subsets.
- \(Y_j\) is the target subset for child \(j\).
- \(|Y_j|/|Y|\) is the proportion of samples in that child.

For Entropy, this is called information gain. When using Gini impurity, the same calculation measures the reduction in Gini impurity.

The algorithm selects the feature or split that produces the greatest impurity reduction.

### 4. Categorical Feature Splitting

For categorical features, the dataset is divided into subsets based on each unique category.

For example, an `Outlook` feature may create these branches:

- Sunny
- Overcast
- Rain

Each branch contains samples whose `Outlook` value matches that category.

### 5. Numerical Feature Splitting

For numerical features, the algorithm:

1. Sorts the unique feature values.
2. Calculates candidate thresholds between consecutive values.
3. Splits the dataset into left and right subsets.
4. Calculates the weighted impurity of each candidate split.
5. Selects the threshold that produces the greatest impurity reduction.

For a threshold \(t\):

$$
X_{\text{left}}: x\leq t
$$

$$
X_{\text{right}}: x>t
$$

This allows the algorithm to find useful split points for continuous numerical features.

### 6. Recursive Tree Construction

After selecting the best split, the algorithm recursively builds subtrees for the resulting subsets.

Tree construction stops when a stopping condition is reached, such as:

- All samples in a node belong to the same class.
- No usable features remain.
- The maximum depth is reached.
- The node does not contain enough samples to split.

When further splitting is not possible, the algorithm can predict the majority class of the current node.

## 🛠️ Implementation

The project contains the following core functions:

| Function | Purpose |
|---|---|
| `compute_entropy(y)` | Calculates entropy for the target labels. |
| `gini(y)` | Calculates Gini impurity. |
| `calculate_impurity(y, criterion)` | Calculates impurity using the selected criterion. |
| `split_dataset(X, y, feature, feature_type, criterion)` | Splits the dataset using categorical values or numerical thresholds. |
| `information_gain(X, y, feature, feature_type, criterion)` | Calculates the impurity reduction from a feature split. |
| `best_feature(X, y, feature_types, criterion)` | Selects the feature that provides the greatest impurity reduction. |
| `detect_feature_types(X)` | Identifies numerical and categorical features. |
| `decision_tree(...)` | Recursively constructs the decision tree. |
| `predict(tree, sample, features, feature_type)` | Predicts a class by traversing the constructed tree. |

## 📊 Experiments and Results

The implementation was evaluated on three datasets and compared with Scikit-learn's `DecisionTreeClassifier`.

### Accuracy Comparison

| Dataset | From Scratch | Scikit-learn |
|---|---:|---:|
| Play Tennis | 89.31% | 89.31% |
| Breast Cancer | 93.86% | 92.11% |
| Titanic | 72.97% | 72.97% |

### 1. Play Tennis Dataset

Both implementations achieved the same accuracy and produced identical confusion matrices.

**From-Scratch Decision Tree**

```text
[[249 116]
 [  0 720]]
```

**Scikit-learn Decision Tree**

```text
[[249 116]
 [  0 720]]
```

### 2. Breast Cancer Dataset

**From-Scratch Decision Tree**

```text
[[68  0]
 [ 7 39]]
```

**Scikit-learn Decision Tree**

```text
[[67  1]
 [ 8 38]]
```

The from-scratch implementation achieved higher accuracy in this experiment.

### 3. Titanic Dataset

**From-Scratch Decision Tree**

```text
[[13  1]
 [ 9 14]]
```

**Scikit-learn Decision Tree**

```text
[[11  3]
 [ 7 16]]
```

Both implementations achieved the same accuracy despite producing different confusion matrices. This demonstrates that models can have identical overall accuracy while making different predictions.

> **Note:** These results represent the reported experiments and their respective train-test splits. The standard Play Tennis teaching dataset contains only 14 examples, so verify the dataset's identity and the evaluation setup before interpreting its 1,085-sample confusion matrix.

## ⚖️ Comparison with Scikit-learn

The purpose of this comparison is to evaluate the behavior of a custom implementation against a widely used machine learning library.

Differences in tree structure and predictions can arise from:

- How categorical features are represented and split.
- How numerical thresholds are evaluated.
- Which splitting criterion is used.
- Whether features can be reused in different branches.
- Stopping conditions and other hyperparameters.
- Dataset preprocessing and train-test splits.

Therefore, matching accuracy does not necessarily mean the trees have identical structures or make identical predictions on every sample.

## 💻 Technologies Used

- **Python** — Core implementation.
- **NumPy** — Numerical computations and dataset operations.
- **Pandas** — Dataset loading and preprocessing.
- **Scikit-learn** — Baseline decision tree and evaluation metrics.
- **Matplotlib / Seaborn** — Visualization, if used in the notebook.
- **Jupyter Notebook** — Experimentation and implementation.

## 🚀 Getting Started

### Prerequisites

Install Python 3.10 or a compatible version and the required libraries.

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

### Clone the Repository

```bash
git clone https://github.com/AbdulHadi640/Decision-Tree-Classifier-From-Scratch.git
cd Decision-Tree-Classifier-From-Scratch
```

Replace the repository URL and folder name if your actual GitHub repository uses a different name.

### Run the Notebook

```bash
jupyter notebook
```

Open the project notebook and run the cells in order.

## 🎯 Learning Outcomes

Through this project, I explored:

- The mathematical foundations of decision trees.
- Entropy, Gini impurity, and impurity reduction.
- Feature selection using information gain.
- Categorical and numerical data splitting.
- Numerical threshold optimization.
- Recursive algorithms and tree traversal.
- Majority-class prediction and stopping conditions.
- Comparing a custom machine learning algorithm with Scikit-learn.


## 👨‍💻 Author

**Abdul Hadi**

- GitHub: [@AbdulHadi640](https://github.com/AbdulHadi640)
- LinkedIn: [Abdul Hadi](https://www.linkedin.com/in/abdul-hadi-a1b550371/)

---

If you find this project useful for learning how decision trees work internally, feel free to explore the implementation and share your feedback.

**Built to understand machine learning from the inside out.** 🌱