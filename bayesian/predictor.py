import numpy as np

def evaluate(Y_true, Y_pred):
    accuracy = float(np.mean(Y_true == Y_pred))
    return accuracy

def softmax(a):
    # Function to calculate the softmax probabilities
    exp_a = np.exp(a - np.max(a))
    return exp_a / exp_a.sum(axis=0)

def classify(X, W, W0, numtypes):
    # Classify the test data
    A = []
    for i in range(numtypes):
        a_k = np.dot(X, W[i]) + W0[i]
        A.append(a_k)
    A = np.array(A)

    return np.argmax(softmax(A), axis=0)