# Quick Start Guide

## Setup (5 minutes)

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/mnist-neural-network.git
cd mnist-neural-network
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- scipy >= 1.7.0

### 3. Run Training
```bash
cd src
python train.py
```

This will:
1. Download MNIST dataset (~12 MB) to `data/` directory
2. Train for 80 epochs (~6 minutes on CPU)
3. Save results to `results/` directory

## Expected Output

```
Loading MNIST dataset...
✓ Data loaded successfully:
  Training: 54000 samples
  Validation: 6000 samples
  Test: 10000 samples

Creating model...
✓ Model created: 784 → 1024 (ReLU) → 10 (Softmax)
  Total parameters: 814,090
✓ Optimizer: SGD with momentum=0.9, weight_decay=0.0001

============================================================
TRAINING START
============================================================
Epoch   0/80 | Train Loss: 0.4521 | Train Acc: 0.8745 | Val Loss: 0.2156 | Val Acc: 0.9345
Epoch   5/80 | Train Loss: 0.1234 | Train Acc: 0.9612 | Val Loss: 0.1456 | Val Acc: 0.9556
...
Epoch  75/80 | Train Loss: 0.0456 | Train Acc: 0.9634 | Val Loss: 0.1123 | Val Acc: 0.9445
============================================================
TRAINING COMPLETE
Best Validation Accuracy: 0.9445
============================================================

============================================================
FINAL EVALUATION
============================================================

Test Loss: 0.1134
Test Accuracy: 0.9425 (94.25%)

Per-Digit Accuracy:
  Digit 0: 0.9714 (97.14%)
  Digit 1: 0.9813 (98.13%)
  Digit 2: 0.9254 (92.54%)
  Digit 3: 0.9366 (93.66%)
  Digit 4: 0.9188 (91.88%)
  Digit 5: 0.9329 (93.29%)
  Digit 6: 0.9124 (91.24%)
  Digit 7: 0.9456 (94.56%)
  Digit 8: 0.9234 (92.34%)
  Digit 9: 0.9421 (94.21%)

============================================================

✓ Saved model to results/model_final.pkl
✓ Saved training history to results/training_history.json
✓ Saved predictions to results/predictions.npy
✓ Saved training curves to results/training_curves.png

✓ Training complete! All results saved to 'results/' directory.
```

## Testing Gradients

Before training, you can verify gradient implementation:

```bash
cd src
python gradient_check.py
```

Expected output:
```
============================================================
GRADIENT CHECKING
============================================================

Parameter 0: (784, 64)
  Max relative error: 2.34e-08
  Status: ✓ PASS

Parameter 1: (1, 64)
  Max relative error: 1.45e-08
  Status: ✓ PASS

Parameter 2: (64, 10)
  Max relative error: 3.12e-08
  Status: ✓ PASS

Parameter 3: (1, 10)
  Max relative error: 1.89e-08
  Status: ✓ PASS

============================================================
✓ ALL GRADIENTS CORRECT
============================================================
```

## Using the Trained Model

```python
import numpy as np
import pickle
from network import NeuralNetwork

# Load model
with open('results/model_final.pkl', 'rb') as f:
    weights = pickle.load(f)

model = NeuralNetwork(784, 1024, 10)
model.fc1.W = weights['fc1_W']
model.fc1.b = weights['fc1_b']
model.fc2.W = weights['fc2_W']
model.fc2.b = weights['fc2_b']

# Make predictions
predictions = model.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.4f}")
```

## Customizing Training

Edit `train.py` to change:

```python
# Model architecture
model = NeuralNetwork(input_dim=784, hidden_dim=1024, output_dim=10)

# Training hyperparameters
optimizer = SGD(learning_rate=0.01, momentum=0.9, weight_decay=0.0001)

# Learning rate schedule
lr_schedule = {
    30: 0.005,
    50: 0.001,
    65: 0.0005
}

# Training duration
history = train(..., epochs=80, batch_size=128, ...)
```

## Troubleshooting

### Issue: Download fails
**Solution:** The script tries multiple mirrors. If all fail, manually download from:
- https://storage.googleapis.com/tensorflow/tf-keras-datasets/
- http://yann.lecun.com/exdb/mnist/

### Issue: Out of memory
**Solution:** Reduce batch size in `train.py`:
```python
history = train(..., batch_size=64, ...)  # Instead of 128
```

### Issue: Training too slow
**Solution:** Reduce number of epochs:
```python
history = train(..., epochs=40, ...)  # Instead of 80
```

### Issue: Accuracy lower than expected
**Solution:** Make sure:
1. Random seed is set: `np.random.seed(42)`
2. Learning rate schedule is enabled
3. Training for full 80 epochs

## Next Steps

1. Modify architecture in `src/network.py`
2. Implement new optimizers in `src/optimizer.py`
3. Add data augmentation in `src/data_utils.py`
4. Experiment with different hyperparameters

## Getting Help

- Check `docs/` for detailed documentation
- Open an issue on GitHub
- Read the source code - all files are well-commented!
