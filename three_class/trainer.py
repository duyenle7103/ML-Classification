import numpy as np

def pseudo_inverse(x):
    return np.linalg.pinv(x)

def train_classifier(X_train, Y_train):
    # Add bias term to the input data
    X_tilde = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

    # Compute pseudo-inverse and T
    X_pseudo_inv = pseudo_inverse(X_tilde)
    T = Y_train

    # Compute parameters matrix W
    W = np.dot(X_pseudo_inv, T)

    return W