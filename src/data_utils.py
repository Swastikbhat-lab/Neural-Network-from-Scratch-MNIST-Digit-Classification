"""
Utilities for loading and preprocessing MNIST data.
"""

import numpy as np
import gzip
import os
import urllib.request


def download_mnist(data_dir='./data'):
    """
    Download MNIST dataset if not already present.
    
    Args:
        data_dir: Directory to save data
    """
    os.makedirs(data_dir, exist_ok=True)
    
    base_url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]
    
    for file in files:
        filepath = os.path.join(data_dir, file)
        if not os.path.exists(filepath):
            print(f"Downloading {file}...")
            try:
                urllib.request.urlretrieve(base_url + file, filepath)
                print(f"✓ Downloaded {file}")
            except Exception as e:
                print(f"Error downloading {file}: {e}")
                # Try alternative source
                alt_url = f'http://yann.lecun.com/exdb/mnist/{file}'
                try:
                    urllib.request.urlretrieve(alt_url, filepath)
                    print(f"✓ Downloaded {file} from alternative source")
                except:
                    raise Exception(f"Failed to download {file}")
        else:
            print(f"✓ {file} already exists")


def load_mnist_images(filename):
    """
    Load MNIST images from gzip file.
    
    Args:
        filename: Path to gzip file
    
    Returns:
        Images as numpy array of shape (num_images, 784)
    """
    with gzip.open(filename, 'rb') as f:
        # Read header
        magic = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        
        # Read image data
        buf = f.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows * cols)
        
        # Normalize to [0, 1]
        data = data.astype(np.float32) / 255.0
        
        return data


def load_mnist_labels(filename):
    """
    Load MNIST labels from gzip file.
    
    Args:
        filename: Path to gzip file
    
    Returns:
        Labels as numpy array of shape (num_labels,)
    """
    with gzip.open(filename, 'rb') as f:
        # Read header
        magic = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')
        
        # Read label data
        buf = f.read(num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8)
        
        return labels


def load_mnist(data_dir='./data', download=True):
    """
    Load MNIST dataset.
    
    Args:
        data_dir: Directory containing data
        download: Whether to download if not present
    
    Returns:
        X_train: Training images (54000, 784)
        y_train: Training labels (54000,)
        X_val: Validation images (6000, 784)
        y_val: Validation labels (6000,)
        X_test: Test images (10000, 784)
        y_test: Test labels (10000,)
    """
    if download:
        download_mnist(data_dir)
    
    # Load training data
    X_train_full = load_mnist_images(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'))
    y_train_full = load_mnist_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
    
    # Load test data
    X_test = load_mnist_images(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'))
    y_test = load_mnist_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))
    
    # Split training into train and validation
    split_idx = 54000
    X_train = X_train_full[:split_idx]
    y_train = y_train_full[:split_idx]
    X_val = X_train_full[split_idx:]
    y_val = y_train_full[split_idx:]
    
    print(f"\n✓ Data loaded successfully:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def get_batches(X, y, batch_size=128, shuffle=True):
    """
    Generate batches from data.
    
    Args:
        X: Input data
        y: Labels
        batch_size: Batch size
        shuffle: Whether to shuffle data
    
    Yields:
        Batches of (X_batch, y_batch)
    """
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        
        yield X[batch_indices], y[batch_indices]
