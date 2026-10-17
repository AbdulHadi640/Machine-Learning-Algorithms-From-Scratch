# Decision Tree Classifier From Scratch 🌳

A Decision Tree Classifier implemented from scratch using **Python and NumPy**, supporting Entropy, Gini Impurity, categorical and numerical features, recursive tree construction, configurable stopping conditions, prediction, and tree visualization using Graphviz.

The implementation is evaluated against Scikit-learn's `DecisionTreeClassifier` on three datasets: Play Tennis, Breast Cancer, and Titanic.

## 📌 Project Overview

Decision trees are supervised machine learning algorithms that make predictions by recursively splitting data according to feature values.

This project implements the core decision tree algorithm from scratch to explore how it selects features, evaluates candidate splits, constructs a tree, and predicts target classes.

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

Entropy measures the uncertainty or impurity of a dataset. It is used to evaluate how mixed the target classes are within a node.

The entropy of a dataset is defined as:

$$
H(Y) = -\sum_{i=1}^{C} p_i \log_2(p_i)
$$

Where:

- \(C\) is the number of target classes.
- \(p_i\) is the proportion of samples belonging to class \(i\).

A pure node has zero entropy because all samples belong to the same class.

### 2. Gini Impurity

Gini impurity measures the probability of incorrectly classifying a randomly selected sample if its class is predicted according to the class distribution within the node.

It is calculated as:

$$
G(Y) = 1 - \sum_{i=1}^{C} p_i^2
$$

A lower Gini impurity indicates a purer node. A pure node has a Gini impurity of zero.

### 3. Impurity Reduction

The algorithm evaluates candidate splits by comparing the parent's impurity with the weighted impurity of its child nodes.

The impurity reduction is:

$$
\Delta I = I(Y) -
\sum_{j=1}^{K}
\frac{|Y_j|}{|Y|} I(Y_j)
$$

Where:

- \(I(Y)\) is the impurity of the parent node.
- \(K\) is the number of child subsets.
- \(|Y|\) is the number of samples in the parent node.
- \(|Y_j|\) is the number of samples in child subset \(j\).
- \(I(Y_j)\) is the impurity of child subset \(j\).

The split producing the greatest impurity reduction is selected.

For Entropy, this reduction is commonly called **Information Gain**.

### 4. Categorical Feature Splitting

Categorical features are divided into subsets based on their unique values.

For example, an `Outlook` feature may create branches for:

- Sunny
- Overcast
- Rain

Each branch contains the samples matching the corresponding category.

### 5. Numerical Feature Splitting

For numerical features, the algorithm evaluates candidate thresholds to divide observations into two subsets.

Given a threshold \(t\), the split is:

$$
X_{\text{left}}: x \leq t
$$

$$
X_{\text{right}}: x > t
$$

Candidate thresholds can be generated from the midpoints between sorted unique feature values:

$$
t_i = \frac{x_i + x_{i+1}}{2}
$$

The algorithm evaluates these candidate splits and selects the threshold with the greatest impurity reduction.

### 6. Recursive Tree Construction

The algorithm recursively builds subtrees using the resulting data subsets until a stopping condition is reached.

Possible stopping conditions include:

- All samples belong to the same class.
- No usable features remain.
- The maximum depth is reached.
- There are insufficient samples to split.
- No valid split can improve the node.

When further splitting is not possible, the node can predict the majority class of its samples.

## 🌳 Decision Tree Visualization

The project includes a Graphviz-based visualization function for displaying the learned decision tree structure.

The visualization represents:

- **Decision nodes:** Features selected for splitting.
- **Numerical thresholds:** Values used to divide numerical features.
- **Edges:** Branch conditions connecting parent and child nodes.
- **Leaf nodes:** Final predicted classes.

### Generate and Save a Tree

After training the classifier and obtaining the learned tree:

```python
graph = visualize_tree(tree)

graph.render(
    "decision_tree",
    format="png",
    cleanup=True
)

graph
```

Each dataset's tree can be visualized and saved separately:

```python
visualize_tree(play_tennis_tree).render(
    "play_tennis",
    format="png",
    cleanup=True
)

visualize_tree(breast_cancer_tree).render(
    "breast_cancer",
    format="png",
    cleanup=True
)

visualize_tree(titanic_tree).render(
    "titanic",
    format="png",
    cleanup=True
)
```

Replace the tree variable names with the corresponding variables in your notebook.

**Note:** Graphviz requires both the Python package and the Graphviz system application. The Graphviz executable must be available through your system's `PATH`.

## 📊 Experiments and Results

The from-scratch classifier was evaluated against Scikit-learn's `DecisionTreeClassifier` using accuracy and confusion matrices.

### Accuracy Comparison

| Dataset | From Scratch | Scikit-learn |
|---|---:|---:|
| Play Tennis | 89.40% | 89.40% |
| Breast Cancer | 92.11% | 92.11% |
| Titanic | 78.38% | 78.38% |

### 1. Play Tennis

Both implementations achieved an accuracy of **89.40%**.

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

Both implementations achieved an accuracy of **92.11%**.

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

Both models achieved the same overall accuracy but produced different confusion matrices, indicating differences in their individual predictions.

### 3. Titanic

Both implementations achieved an accuracy of **78.38%**.

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

> **Important:** These results reflect the reported evaluation runs and their respective dataset splits. The Play Tennis confusion matrix totals 1,085 samples, which is unusual for the standard 14-row Play Tennis dataset. Verify the dataset identity, sample counts, and evaluation setup before publishing these results.

## ⚖️ Comparison with Scikit-learn

The comparison evaluates the custom implementation against an established machine learning library.

Differences in tree structures and predictions can result from:

- Feature representation and categorical splitting.
- Numerical threshold selection.
- Splitting criteria.
- Feature reuse across branches.
- Stopping conditions and hyperparameters.
- Dataset preprocessing and train-test splits.
- Tie-breaking behavior during split selection.

Matching accuracy does not necessarily mean that two models have identical tree structures or predictions.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| NumPy | Numerical computations and array operations |
| Pandas | Dataset loading and preprocessing |
| Scikit-learn | Baseline model and evaluation metrics |
| Graphviz | Decision tree visualization |
| Jupyter Notebook | Implementation and experimentation |
| Matplotlib / Seaborn | Data visualization, where used |

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or later recommended.
- Jupyter Notebook.
- Graphviz system application for tree visualization.

### 1. Install Python Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn graphviz jupyter
```

### 2. Install Graphviz

Download and install Graphviz from its official website:

[Graphviz Downloads](https://graphviz.org/download/)

Ensure the Graphviz `bin` directory is available through your system `PATH`.

To verify the installation, run:

```bash
dot -V
```

If the command displays the installed Graphviz version, the executable is accessible.

### 3. Clone the Repository

```bash
git clone https://github.com/AbdulHadi640/Decision-Tree-Classifier-From-Scratch.git
cd Decision-Tree-Classifier-From-Scratch
```

Replace the repository URL if your actual repository uses a different name.

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the relevant notebook and execute the cells in order.

## 🎯 Learning Outcomes

Through this project, I explored and implemented:

- Entropy and Gini impurity from mathematical definitions.
- Information gain and impurity reduction.
- Feature selection for categorical and numerical data.
- Candidate threshold generation for numerical features.
- Recursive decision tree construction.
- Stopping conditions and majority-class prediction.
- Prediction through tree traversal.
- Model evaluation using accuracy and confusion matrices.
- Decision tree visualization with Graphviz.
- Comparison of a custom machine learning algorithm against Scikit-learn.

## 👨‍💻 Author

**Abdul Hadi**

BS Computer Science Student | Aspiring AI/ML Engineer

- **GitHub:** [@AbdulHadi640](https://github.com/AbdulHadi640)
- **LinkedIn:** [Abdul Hadi](https://www.linkedin.com/in/abdul-hadi-a1b550371/)

Built from scratch to understand machine learning from the inside out. 🌱
