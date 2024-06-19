import numpy as np
from itertools import combinations

def generate_combinations(numtypes):
    # Create combination of each pair
    elements = range(numtypes)
    comb = list(combinations(elements, 2))

    return comb

def kfold_split(X, Y, n_splits=5, shuffle=True):
    # Get indices
    indices = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(indices)

    # Get integer division
    fold_size = X.shape[0] // n_splits
    for i in range(0, X.shape[0], fold_size):
        test_indices = indices[i:i+fold_size]
        train_indices = np.concatenate((indices[:i], indices[i+fold_size:]))
        yield X[train_indices], Y[train_indices], X[test_indices], Y[test_indices]