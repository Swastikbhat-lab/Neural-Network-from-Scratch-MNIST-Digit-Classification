# Neural Network from Scratch: MNIST Digit Classification

A fully-connected neural network implemented from scratch using only NumPy, achieving **94.25% test accuracy** on MNIST digit classification.

##  Project Overview

This project implements a complete neural network from first principles without using any deep learning frameworks (no TensorFlow, PyTorch, or Keras). All components including forward propagation, backpropagation, and optimization are manually implemented and mathematically verified.

### Key Achievements
-  **94.25% test accuracy** on MNIST
-  All gradients verified with numerical gradient checking
-  Complete from-scratch implementation using only NumPy
-  Comprehensive documentation and analysis

##  Results Summary

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 94.25% |
| **Train Accuracy** | 96.34% |
| **Validation Accuracy** | 94.45% |
| **Train-Val Gap** | 1.89% (excellent generalization) |
| **Best Per-Digit** | 98.1% (Digit 1) |
| **Worst Per-Digit** | 91.2% (Digit 6) |

##  Architecture

```
Input (784) → Dense (1024, ReLU) → Output (10, Softmax)
```

**Parameters:**
- Input neurons: 784 (28×28 flattened images)
- Hidden neurons: 1024 with ReLU activation
- Output neurons: 10 (digit classes 0-9)
- Total parameters: 814,090

**Initialization:** He initialization (optimal for ReLU)

##  Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | SGD with Momentum |
| Learning Rate | 0.01 (with decay) |
| Momentum | 0.9 |
| Weight Decay | 0.0001 (L2 regularization) |
| Batch Size | 128 |
| Epochs | 80 |
| LR Decay | At epochs 30, 50, 65 |
| Gradient Clipping | [-5, 5] |




### 4. Results
Training outputs will be saved to `results/`:
- `model_final.pkl` - Trained model weights
- `training_history.json` - Loss and accuracy per epoch
- `training_curves.png` - Visualization of training
- `confusion_matrix.png` - Classification errors
- `sample_predictions.png` - Example predictions

##  Implementation Details

### Core Components

**1. Dense Layer**
- Forward: `h = X @ W + b`
- Backward: Chain rule gradients
- He initialization: `W ~ N(0, sqrt(2/n_in))`

**2. ReLU Activation**
- Forward: `max(0, x)`
- Backward: `gradient * (x > 0)`

**3. Softmax + Cross-Entropy Loss**
- Combined for numerical stability
- Gradient: `y_pred - y_true`

**4. SGD with Momentum**
- `v = momentum * v - lr * gradient`
- `weights += v`

### Gradient Verification
All gradients verified with numerical gradient checking:
```python
numerical_grad = (loss(w + eps) - loss(w - eps)) / (2 * eps)
relative_error = |analytical - numerical| / (|analytical| + |numerical|)
```
All relative errors < 1e-7 

##  Training History

The model was trained through 6 iterations to reach 94.25%:

| Iteration | Configuration | Result | Issue |
|-----------|--------------|--------|-------|
| 1 | 128 units, LR 0.01 | Gradient errors | NaN loss |
| 2 | 128 units, fixed gradients | 88.1% | Underfitting |
| 3 | 256 units, LR 0.01 | 88.9% | Capacity issue |
| 4 | 512 units, LR 0.01 | 89.2% | Still limited |
| 5 | 1024 units, LR 0.15 | 10% (failure) | LR too high |
| **6** | **1024 units, LR 0.01** | **94.25%**  | **Success!** |

### Key Learnings
- Larger networks need **smaller** learning rates
- Gradient verification is essential (caught bugs early)
- Momentum helps convergence significantly
- 1024 units provided sufficient capacity

##  Visualizations

The project generates several visualizations:

1. **Training Curves**: Loss and accuracy over epochs
2. **Confusion Matrix**: Common misclassification patterns
3. **Sample Predictions**: Visual examples of correct/incorrect predictions
4. **Per-Digit Accuracy**: Performance breakdown by digit
5. **Performance Timeline**: All iterations compared

## Some more details 

- **Learning Rate Decay**: Decay at epochs 30, 50, 65
- **Gradient Clipping**: Prevents exploding gradients
- **He Initialization**: Optimal for ReLU networks
- **Momentum**: Accelerates convergence
- **Weight Decay**: L2 regularization prevents overfitting


##  Comparison to Baselines

| Approach | Accuracy |
|----------|----------|
| Random Guess | 10% |
| Logistic Regression | ~92% |
| **This Project** | **94.25%** |
| Best FC Network (literature) | ~96-97% |
| CNNs (state-of-art) | >99% |

##  Running Tests

Test gradient checking:
```bash
python src/gradient_check.py
```

Expected output: All relative errors < 1e-7

##  Contributing

This is an educational project demonstrating neural network fundamentals. Suggestions and improvements welcome!

##  License

MIT License - feel free to use for educational purposes.

##  Author

**Swastik Bhat**
- Course: CSCI-4364/6364 – Machine Learning
- Semester: Fall 2025

##  Acknowledgments

- MNIST dataset: Yann LeCun et al.
- Inspiration from CS231n (Stanford) and Deep Learning (Goodfellow et al.)
- Mathematical foundations verified against academic literature


