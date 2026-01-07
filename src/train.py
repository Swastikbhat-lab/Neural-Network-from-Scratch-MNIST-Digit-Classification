"""
Main training script for MNIST neural network.
"""

import numpy as np
import json
import pickle
import os
from network import NeuralNetwork
from optimizer import SGD
from data_utils import load_mnist, get_batches
import matplotlib.pyplot as plt


def train(model, optimizer, X_train, y_train, X_val, y_val, 
          epochs=80, batch_size=128, lr_schedule=None):
    """
    Train neural network.
    
    Args:
        model: Neural network
        optimizer: Optimizer
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs: Number of training epochs
        batch_size: Batch size
        lr_schedule: Dict mapping epoch to learning rate
    
    Returns:
        history: Training history
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []
    }
    
    best_val_acc = 0
    
    print("\n" + "="*60)
    print("TRAINING START")
    print("="*60)
    
    for epoch in range(epochs):
        # Learning rate scheduling
        if lr_schedule and epoch in lr_schedule:
            new_lr = lr_schedule[epoch]
            optimizer.set_learning_rate(new_lr)
            print(f"\n→ Learning rate changed to {new_lr}")
        
        # Training
        train_losses = []
        train_accs = []
        
        for X_batch, y_batch in get_batches(X_train, y_train, batch_size):
            loss, acc = model.train_step(X_batch, y_batch)
            optimizer.step(model.get_params(), model.get_grads())
            
            train_losses.append(loss)
            train_accs.append(acc)
        
        # Validation
        val_loss, val_acc = model.evaluate(X_val, y_val)
        
        # Record history
        history['train_loss'].append(np.mean(train_losses))
        history['train_acc'].append(np.mean(train_accs))
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(optimizer.lr)
        
        # Print progress
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {history['train_loss'][-1]:.4f} | "
                  f"Train Acc: {history['train_acc'][-1]:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    print("\n" + "="*60)
    print(f"TRAINING COMPLETE")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print("="*60 + "\n")
    
    return history


def plot_training_history(history, save_path='results/training_curves.png'):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved training curves to {save_path}")
    plt.close()


def evaluate_model(model, X_test, y_test):
    """Evaluate model and print detailed results."""
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    # Overall accuracy
    test_loss, test_acc = model.evaluate(X_test, y_test)
    predictions = model.predict(X_test)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Per-class accuracy
    print("\nPer-Digit Accuracy:")
    for digit in range(10):
        mask = y_test == digit
        digit_acc = np.mean(predictions[mask] == y_test[mask])
        print(f"  Digit {digit}: {digit_acc:.4f} ({digit_acc*100:.2f}%)")
    
    print("\n" + "="*60 + "\n")
    
    return test_acc, predictions


def save_results(model, history, predictions, save_dir='results'):
    """Save model and training results."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_dir, 'model_final.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'fc1_W': model.fc1.W,
            'fc1_b': model.fc1.b,
            'fc2_W': model.fc2.W,
            'fc2_b': model.fc2.b
        }, f)
    print(f"✓ Saved model to {model_path}")
    
    # Save history
    history_path = os.path.join(save_dir, 'training_history.json')
    # Convert numpy arrays to lists for JSON
    history_json = {k: [float(v) for v in vals] for k, vals in history.items()}
    with open(history_path, 'w') as f:
        json.dump(history_json, f, indent=2)
    print(f"✓ Saved training history to {history_path}")
    
    # Save predictions
    pred_path = os.path.join(save_dir, 'predictions.npy')
    np.save(pred_path, predictions)
    print(f"✓ Saved predictions to {pred_path}")


def main():
    """Main training function."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load data
    print("\nLoading MNIST dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()
    
    # Create model
    print("\nCreating model...")
    model = NeuralNetwork(input_dim=784, hidden_dim=1024, output_dim=10)
    print("✓ Model created: 784 → 1024 (ReLU) → 10 (Softmax)")
    print(f"  Total parameters: {784*1024 + 1024 + 1024*10 + 10:,}")
    
    # Create optimizer
    optimizer = SGD(learning_rate=0.01, momentum=0.9, weight_decay=0.0001)
    print("✓ Optimizer: SGD with momentum=0.9, weight_decay=0.0001")
    
    # Learning rate schedule
    lr_schedule = {
        30: 0.005,  # Decay at epoch 30
        50: 0.001,  # Decay at epoch 50
        65: 0.0005  # Decay at epoch 65
    }
    
    # Train
    history = train(
        model, optimizer,
        X_train, y_train,
        X_val, y_val,
        epochs=80,
        batch_size=128,
        lr_schedule=lr_schedule
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Final evaluation
    test_acc, predictions = evaluate_model(model, X_test, y_test)
    
    # Save results
    save_results(model, history, predictions)
    
    print("\n✓ Training complete! All results saved to 'results/' directory.\n")


if __name__ == "__main__":
    main()
