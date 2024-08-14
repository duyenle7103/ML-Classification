import numpy as np

def evaluate(Y_true, Y_pred):
    accuracy = np.sum(Y_pred == Y_true) / len(Y_true)
    return accuracy

def predict(X, W):
    # Augment data matrix X with bias term
    X_augmented = np.hstack((1, X)).reshape(-1, 1)
    return W.T.dot(X_augmented)