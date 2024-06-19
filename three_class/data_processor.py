import numpy as np
from itertools import combinations

def one_hot_encoder(Y, numtypes):
    T = np.zeros((Y.shape[0], numtypes))
    for i in range(numtypes):
        T[Y == i, i] = 1
        
    return T

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