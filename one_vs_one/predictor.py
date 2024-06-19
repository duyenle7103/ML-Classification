import numpy as np

def predict(W, X_test):
    # Add bias term to the input data
    X_tilde = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

    # Compute predictions
    Y_pred = np.dot(X_tilde, W)

    return np.where(Y_pred >= 0, 1, -1)

def evaluate(Y_true, Y_pred):
    accuracy = float(np.mean(Y_true == Y_pred))
    return accuracy