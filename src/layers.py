"""
Neural network layers implemented from scratch.
"""

import numpy as np


class Dense:
    """Fully connected (dense) layer with He initialization."""
    
    def __init__(self, input_dim, output_dim):
        """
        Initialize dense layer.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output features
        """
        # He initialization for ReLU networks
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros((1, output_dim))
        
        # Cache for backpropagation
        self.X = None
        self.dW = None
        self.db = None
    
    def forward(self, X):
        """
        Forward pass: h = X @ W + b
        
        Args:
            X: Input of shape (batch_size, input_dim)
        
        Returns:
            Output of shape (batch_size, output_dim)
        """
        self.X = X
        return X @ self.W + self.b
    
    def backward(self, dout):
        """
        Backward pass: compute gradients.
        
        Args:
            dout: Gradient from next layer, shape (batch_size, output_dim)
        
        Returns:
            Gradient w.r.t input, shape (batch_size, input_dim)
        """
        batch_size = self.X.shape[0]
        
        # Gradients
        self.dW = self.X.T @ dout / batch_size
        self.db = np.sum(dout, axis=0, keepdims=True) / batch_size
        dX = dout @ self.W.T
        
        return dX
    
    def get_params(self):
        """Return list of parameters [W, b]."""
        return [self.W, self.b]
    
    def get_grads(self):
        """Return list of gradients [dW, db]."""
        return [self.dW, self.db]


class ReLU:
    """ReLU activation function."""
    
    def __init__(self):
        self.X = None
    
    def forward(self, X):
        """
        Forward pass: max(0, X)
        
        Args:
            X: Input of shape (batch_size, features)
        
        Returns:
            Output of same shape as X
        """
        self.X = X
        return np.maximum(0, X)
    
    def backward(self, dout):
        """
        Backward pass.
        
        Args:
            dout: Gradient from next layer
        
        Returns:
            Gradient w.r.t input
        """
        return dout * (self.X > 0)
