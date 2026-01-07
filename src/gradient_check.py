"""
Numerical gradient checking for verification.
"""

import numpy as np
from network import NeuralNetwork
from loss import softmax_cross_entropy_loss


def numerical_gradient(f, x, eps=1e-5):
    """
    Compute numerical gradient using finite differences.
    
    Args:
        f: Function that takes x and returns scalar
        x: Input array
        eps: Small perturbation
    
    Returns:
        Numerical gradient of same shape as x
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        old_value = x[idx]
        
        # f(x + eps)
        x[idx] = old_value + eps
        pos = f(x)
        
        # f(x - eps)
        x[idx] = old_value - eps
        neg = f(x)
        
        # Gradient: (f(x+eps) - f(x-eps)) / (2*eps)
        grad[idx] = (pos - neg) / (2 * eps)
        
        # Restore
        x[idx] = old_value
        it.iternext()
    
    return grad


def check_gradients(model, X, y, eps=1e-5):
    """
    Check analytical gradients against numerical gradients.
    
    Args:
        model: Neural network model
        X: Input batch (small)
        y: Labels
        eps: Perturbation for numerical gradient
    
    Returns:
        True if all gradients are correct
    """
    print("\n" + "="*60)
    print("GRADIENT CHECKING")
    print("="*60)
    
    # Forward and backward pass
    logits = model.forward(X)
    loss, dlogits = softmax_cross_entropy_loss(logits, y)
    model.backward(dlogits)
    
    # Get analytical gradients
    params = model.get_params()
    grads = model.get_grads()
    
    all_correct = True
    
    for i, (param, grad) in enumerate(zip(params, grads)):
        # Define loss function for this parameter
        def loss_fn(p):
            # Temporarily replace parameter
            old_param = param.copy()
            param[:] = p
            
            # Forward pass
            logits = model.forward(X)
            loss, _ = softmax_cross_entropy_loss(logits, y)
            
            # Restore
            param[:] = old_param
            
            return loss
        
        # Compute numerical gradient
        numerical_grad = numerical_gradient(loss_fn, param, eps)
        
        # Compute relative error
        diff = np.abs(grad - numerical_grad)
        norm = np.abs(grad) + np.abs(numerical_grad)
        relative_error = np.max(diff / (norm + 1e-8))
        
        # Check if correct
        is_correct = relative_error < 1e-5
        status = "✓ PASS" if is_correct else "✗ FAIL"
        
        print(f"\nParameter {i}: {param.shape}")
        print(f"  Max relative error: {relative_error:.2e}")
        print(f"  Status: {status}")
        
        if not is_correct:
            all_correct = False
            print(f"  Sample analytical: {grad.flatten()[:5]}")
            print(f"  Sample numerical:  {numerical_grad.flatten()[:5]}")
    
    print("\n" + "="*60)
    if all_correct:
        print("✓ ALL GRADIENTS CORRECT")
    else:
        print("✗ SOME GRADIENTS INCORRECT")
    print("="*60 + "\n")
    
    return all_correct


if __name__ == "__main__":
    # Set random seed
    np.random.seed(42)
    
    # Create small test case
    model = NeuralNetwork(input_dim=784, hidden_dim=64, output_dim=10)
    X = np.random.randn(5, 784)
    y = np.array([0, 1, 2, 3, 4])
    
    # Check gradients
    check_gradients(model, X, y)
