# Decision Tree Classifier From Scratch 🌳

A Decision Tree Classifier implemented from scratch using Python and NumPy, supporting **Entropy and Gini Impurity**, categorical and numerical features, recursive tree construction, configurable stopping conditions, and decision tree visualization using Graphviz.

The implementation is evaluated against Scikit-learn's `DecisionTreeClassifier` on three datasets: Play Tennis, Breast Cancer, and Titanic.

## 📌 Project Overview

Decision trees are supervised machine learning algorithms that make predictions by recursively splitting data according to feature values.

This project implements the core decision tree algorithm from scratch to explore how it selects features, evaluates splits, constructs a tree, and predicts target classes.

## ✨ Features

- **From-scratch implementation:** Core tree-building and prediction logic implemented using Python and NumPy.
- **Entropy:** Calculates entropy and information gain for split selection.
- **Gini Impurity:** Supports Gini-based impurity reduction.
- **Categorical features:** Creates branches for unique categorical values.
- **Numerical features:** Evaluates candidate thresholds for binary splits.
- **Automatic feature selection:** Selects splits according to the chosen impurity criterion.
- **Recursive tree construction:** Builds subtrees from the resulting data subsets.
- **Configurable stopping conditions:** Supports maximum depth and minimum samples per split, where configured.
- **Majority-class prediction:** Handles nodes where further splitting is not possible.
- **Prediction:** Traverses the learned tree to classify samples.
- **Tree visualization:** Visualizes the learned decision tree using Graphviz.
- **Model evaluation:** Compares accuracy and confusion matrices against Scikit-learn.

## 🧠 How It Works

### 1. Entropy

Entropy measures the uncertainty or impurity of a dataset.

\[
H(Y)=-\sum_{i=1}^{C}p_i\log_2(p_i)
\]

Here, \(p_i\) is the proportion of samples belonging to class \(i\). A pure node has zero entropy.

### 2. Gini Impurity

Gini impurity measures the impurity of a node based on its class distribution.

\[
G(Y)=1-\sum_{i=1}^{C}p_i^2
\]

A lower Gini impurity indicates a purer node.

### 3. Impurity Reduction

The algorithm evaluates candidate splits by comparing the parent's impurity with the weighted impurity of its child nodes.

\[
\Delta I=I(Y)-\sum_{j=1}^{K}\frac{|Y_j|}{|Y|}I(Y_j)
\]

Here, \(I\) represents either entropy or Gini impurity.

The split producing the greatest impurity reduction is selected.

### 4. Categorical Feature Splitting

Categorical features are divided into subsets based on their unique values.

For example, an `Outlook` feature may create branches for `Sunny`, `Overcast`, and `Rain`.

### 5. Numerical Feature Splitting

For numerical features, the algorithm evaluates candidate thresholds to divide observations into two subsets:

\[
X_{\text{left}}: x\leq t
\]

\[
X_{\text{right}}: x>t
\]

The threshold with the greatest impurity reduction is selected.

### 6. Recursive Tree Construction

The algorithm recursively builds subtrees until a stopping condition is reached, such as:

- All samples belong to the same class.
- No usable features remain.
- Maximum depth is reached.
- There are insufficient samples to split.

When further splitting is not possible, the node can predict the majority class.

## 🌳 Decision Tree Visualization

The project includes a Graphviz-based visualization function for displaying the learned tree structure.

The visualization represents:

- **Decision nodes:** Features selected for splitting.
- **Thresholds:** Numerical split values.
- **Edges:** Branch conditions.
- **Leaf nodes:** Predicted classes.

### Generate and Save a Tree

```python
graph = visualize_tree(tree)

graph.render(
    "decision_tree",
    format="png",
    cleanup=True
)

graph
```

Each dataset's tree can be visualized and saved separately, for example:

- `play_tennis.png`
- `breast_cancer.png`
- `titanic.png`

The generated images can be included in the repository to demonstrate how the classifier makes decisions.

## 📊 Experiments and Results

The from-scratch classifier was evaluated against Scikit-learn's `DecisionTreeClassifier`.

### Accuracy Comparison

| Dataset | From Scratch | Scikit-learn |
|---|---:|---:|
| Play Tennis | 89.40% | 89.40% |
| Breast Cancer | 92.11% | 92.11% |
| Titanic | 78.38% | 78.38% |

### 1. Play Tennis

Both implementations achieved **89.40% accuracy**.

**From-Scratch Confusion Matrix**

```text
[[250 115]
 [  0 720]]
```

**Scikit-learn Confusion Matrix**

```text
[[250 115]
 [  0 720]]
```

Both implementations produced identical confusion matrices in this evaluation.

### 2. Breast Cancer

Both implementations achieved **92.11% accuracy**.

**From-Scratch Confusion Matrix**

```text
[[57  3]
 [ 6 48]]
```

**Scikit-learn Confusion Matrix**

```text
[[59  1]
 [ 8 46]]
```

Both models achieved the same accuracy but produced different confusion matrices, indicating differences in their individual predictions.

### 3. Titanic

Both implementations achieved **78.38% accuracy**.

**From-Scratch Confusion Matrix**

```text
[[14  3]
 [ 5 15]]
```

**Scikit-learn Confusion Matrix**

```text
[[10  7]
 [ 1 19]]
```

Despite matching overall accuracy, the models made different predictions and distributed their classification errors differently.

> **Note:** These results reflect the reported evaluation runs and their respective dataset splits. The Play Tennis confusion matrix totals 1,085 samples, so verify the dataset identity and evaluation setup before publication.

## ⚖️ Comparison with Scikit-learn

The comparison evaluates the custom implementation against an established machine learning library.

Differences in tree structures and predictions can result from:

- Feature representation and categorical splitting.
- Numerical threshold selection.
- Splitting criteria.
- Feature reuse across branches.
- Stopping conditions and hyperparameters.
- Dataset preprocessing and train-test splits.

Matching accuracy does not necessarily mean that two models have identical structures or predictions.

## 🛠️ Technologies Used

- **Python** — Core implementation.
- **NumPy** — Numerical computations and array operations.
- **Pandas** — Dataset loading and preprocessing.
- **Scikit-learn** — Baseline model and evaluation metrics.
- **Graphviz** — Decision tree visualization.
- **Jupyter Notebook** — Implementation and experimentation.

## 🚀 Getting Started

### Install Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn graphviz jupyter
```

Install the Graphviz system application separately from:

https://graphviz.org/download/

Ensure the Graphviz `bin` directory is available through your system `PATH`.

### Clone the Repository

```bash
git clone https://github.com/AbdulHadi640/Decision-Tree-Classifier-From-Scratch.git
cd Decision-Tree-Classifier-From-Scratch
```

Replace the repository URL if your actual repository uses a different name.

### Run the Notebook

```bash
jupyter notebook
```

Open the relevant notebook and execute the cells in order.

## 🎯 Learning Outcomes

- Understanding entropy and Gini impurity.
- Implementing impurity reduction and feature selection.
- Splitting categorical and numerical features.
- Finding numerical split thresholds.
- Building recursive decision trees.
- Implementing prediction through tree traversal.
- Evaluating models with accuracy and confusion matrices.
- Visualizing learned decision trees.
- Comparing a custom machine learning algorithm with Scikit-learn.

## 👨‍💻 Author

**Abdul Hadi**

BS Computer Science Student | Aspiring AI/ML Engineer

- GitHub: [@AbdulHadi640](https://github.com/AbdulHadi640)
- LinkedIn: [Abdul Hadi](https://www.linkedin.com/in/abdul-hadi-a1b550371/)

---

**Built from scratch to understand machine learning from the inside out.** 🌱
