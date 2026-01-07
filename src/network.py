"""
Neural network model.
"""

import numpy as np
from layers import Dense, ReLU
from loss import softmax_cross_entropy_loss, compute_accuracy


class NeuralNetwork:
    """Two-layer neural network for MNIST classification."""
    
    def __init__(self, input_dim=784, hidden_dim=1024, output_dim=10):
        """
        Initialize neural network.
        
        Args:
            input_dim: Input dimension (784 for MNIST)
            hidden_dim: Hidden layer size
            output_dim: Number of classes (10 for MNIST)
        """
        self.fc1 = Dense(input_dim, hidden_dim)
        self.relu = ReLU()
        self.fc2 = Dense(hidden_dim, output_dim)
    
    def forward(self, X):
        """
        Forward pass through network.
        
        Args:
            X: Input of shape (batch_size, 784)
        
        Returns:
            Logits of shape (batch_size, 10)
        """
        h = self.fc1.forward(X)
        h = self.relu.forward(h)
        logits = self.fc2.forward(h)
        return logits
    
    def backward(self, dlogits):
        """
        Backward pass through network.
        
        Args:
            dlogits: Gradient from loss, shape (batch_size, 10)
        """
        dx = self.fc2.backward(dlogits)
        dx = self.relu.backward(dx)
        dx = self.fc1.backward(dx)
        return dx
    
    def get_params(self):
        """Get all parameters as a list."""
        return self.fc1.get_params() + self.fc2.get_params()
    
    def get_grads(self):
        """Get all gradients as a list."""
        return self.fc1.get_grads() + self.fc2.get_grads()
    
    def train_step(self, X, y):
        """
        Single training step.
        
        Args:
            X: Input batch of shape (batch_size, 784)
            y: Labels of shape (batch_size,)
        
        Returns:
            loss: Scalar loss value
            accuracy: Batch accuracy
        """
        # Forward pass
        logits = self.forward(X)
        
        # Compute loss and gradient
        loss, dlogits = softmax_cross_entropy_loss(logits, y)
        
        # Backward pass
        self.backward(dlogits)
        
        # Compute accuracy
        accuracy = compute_accuracy(logits, y)
        
        return loss, accuracy
    
    def evaluate(self, X, y):
        """
        Evaluate model on data.
        
        Args:
            X: Input of shape (num_samples, 784)
            y: Labels of shape (num_samples,)
        
        Returns:
            loss: Average loss
            accuracy: Accuracy
        """
        logits = self.forward(X)
        loss, _ = softmax_cross_entropy_loss(logits, y)
        accuracy = compute_accuracy(logits, y)
        return loss, accuracy
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Input of shape (num_samples, 784)
        
        Returns:
            Predicted labels of shape (num_samples,)
        """
        logits = self.forward(X)
        return np.argmax(logits, axis=1)
