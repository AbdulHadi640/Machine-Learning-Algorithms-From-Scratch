# MNIST Neural Network from Scratch (NumPy)

A multi-layer feedforward neural network built entirely from scratch using **Python and NumPy**, designed to classify handwritten digits from the MNIST dataset[cite: 3]. This project implements core deep learning mechanics without relying on high-level framework layers for training logic.

---

## 🚀 Key Features
* **Built from Scratch**: Manual implementation of forward propagation, loss calculation, backward propagation, and parameter updates[cite: 3].
* **Architecture**: 3-layer neural network (`784 -> 25 -> 15 -> 10`) utilizing **ReLU** hidden activations and **Softmax** output classification[cite: 3].
* **He Initialization**: Custom weight scaling ($\text{randn} \times \sqrt{2.0 / n\_row}$) to prevent vanishing/exploding gradients[cite: 3].
* **Mini-Batch Gradient Descent**: Optimized training loop featuring dataset shuffling, dynamic batch slicing, and **Learning Rate Decay**[cite: 3].
* **Framework Benchmark**: Compared side-by-side with a equivalent TensorFlow/Keras Sequential model using the Adam optimizer[cite: 3].

---

## 📊 Performance & Results
* **Custom NumPy Model**: Achieved **96.20% Test Accuracy** using mini-batch gradient descent with learning rate decay[cite: 3].
* **TensorFlow Keras Baseline**: Achieved **96.27% Accuracy** over 5 epochs using industry-standard framework tooling[cite: 3].
* **Conclusion**: Proves that a custom scratch-built NumPy neural network matches the precision of production-grade frameworks[cite: 3].

---

## 🛠️ Tech Stack
* **Language**: Python 3[cite: 3]
* **Core Libraries**: NumPy[cite: 3], TensorFlow / Keras (for dataset loading and benchmarking)[cite: 3]
* **Dataset**: MNIST (60,000 training images, 10,000 test images)[cite: 3]

---

## 📂 Project Structure
1. **Libraries & Setup**: Importing dependencies[cite: 3].
2. **Data Processing**: Loading, flattening ($28 \times 28 \to 784$), and normalizing pixel values (`[0, 1]`)[cite: 3].
3. **One-Hot Encoding**: Transforming integer labels into categorical vectors[cite: 3].
4. **Initialization**: He weight initialization and zero-bias setup[cite: 3].
5. **Loss & Activations**: Categorical Cross-Entropy, Softmax, and ReLU implementations[cite: 3].
6. **Propagation**: Forward and backward pass math derived and coded manually[cite: 3].
7. **Training & Evaluation**: Mini-batch training with learning rate decay and test set accuracy evaluation[cite: 3].
8. **Framework Comparison**: Benchmark validation using Keras Sequential Dense layers[cite: 3].