"""
Optimization algorithms for neural network training.
"""

import numpy as np


class SGD:
    """Stochastic Gradient Descent with momentum."""
    
    def __init__(self, learning_rate=0.01, momentum=0.9, weight_decay=0.0):
        """
        Initialize SGD optimizer.
        
        Args:
            learning_rate: Learning rate
            momentum: Momentum coefficient
            weight_decay: L2 regularization coefficient
        """
        self.lr = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # Velocity for momentum
        self.velocities = []
    
    def step(self, params, grads):
        """
        Update parameters using gradients.
        
        Args:
            params: List of parameter arrays [W1, b1, W2, b2, ...]
            grads: List of gradient arrays [dW1, db1, dW2, db2, ...]
        """
        # Initialize velocities on first call
        if len(self.velocities) == 0:
            self.velocities = [np.zeros_like(p) for p in params]
        
        # Update each parameter
        for i, (param, grad) in enumerate(zip(params, grads)):
            # Add weight decay (L2 regularization)
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param
            
            # Clip gradients to prevent explosion
            grad = np.clip(grad, -5, 5)
            
            # Update velocity: v = momentum * v - lr * grad
            self.velocities[i] = self.momentum * self.velocities[i] - self.lr * grad
            
            # Update parameters: param += v
            param += self.velocities[i]
    
    def set_learning_rate(self, lr):
        """Update learning rate (for learning rate scheduling)."""
        self.lr = lr
