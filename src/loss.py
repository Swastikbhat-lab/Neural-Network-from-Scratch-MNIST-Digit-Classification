"""
Loss functions for neural network training.
"""

import numpy as np


def softmax(logits):
    """
    Compute softmax in a numerically stable way.
    
    Args:
        logits: Array of shape (batch_size, num_classes)
    
    Returns:
        Probabilities of shape (batch_size, num_classes)
    """
    # Subtract max for numerical stability
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def softmax_cross_entropy_loss(logits, y_true):
    """
    Combined softmax and cross-entropy loss.
    
    Args:
        logits: Raw scores of shape (batch_size, num_classes)
        y_true: True labels of shape (batch_size,) or (batch_size, num_classes)
    
    Returns:
        loss: Scalar loss value
        dlogits: Gradient w.r.t logits, shape (batch_size, num_classes)
    """
    batch_size = logits.shape[0]
    
    # Convert labels to one-hot if needed
    if y_true.ndim == 1:
        num_classes = logits.shape[1]
        y_one_hot = np.zeros((batch_size, num_classes))
        y_one_hot[np.arange(batch_size), y_true] = 1
    else:
        y_one_hot = y_true
    
    # Compute probabilities
    probs = softmax(logits)
    
    # Cross-entropy loss
    # Add small epsilon to prevent log(0)
    loss = -np.sum(y_one_hot * np.log(probs + 1e-8)) / batch_size
    
    # Gradient: probs - y_true
    dlogits = (probs - y_one_hot) / batch_size
    
    return loss, dlogits


def compute_accuracy(logits, y_true):
    """
    Compute classification accuracy.
    
    Args:
        logits: Predictions of shape (batch_size, num_classes)
        y_true: True labels of shape (batch_size,)
    
    Returns:
        Accuracy as float in [0, 1]
    """
    predictions = np.argmax(logits, axis=1)
    
    # Handle one-hot encoded labels
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    
    return np.mean(predictions == y_true)
