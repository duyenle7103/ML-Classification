from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from data_loader import load_iris_data

X_train, y_train, numtypes = load_iris_data("input/train.txt")
X_test, y_test, num = load_iris_data("input/test.txt")

print(X_train)
print(X_train.shape)
print(y_train)
print(y_train.shape)
print(X_test)
print(X_test.shape)
print(y_test)
print(y_test.shape)

# Initialize the number of classes
num_classes = len(np.unique(y_train))

print(num_classes)

# Calculate mean and shared covariance matrix
mu = np.array([np.mean(X_train[y_train == label], axis=0) for label in range(num_classes)])

print(mu)

Sigma = np.mean([np.cov(X_train[y_train == label], rowvar=False) for label in range(num_classes)], axis=0)

print(Sigma)

# Calculate inverse of shared covariance matrix
Sigma_inv = np.linalg.inv(Sigma)

# Calculate priors
priors = [np.mean(y_train == label) for label in range(num_classes)]

# Calculate wk and wk0 for each class k
wk_list = []
wk0_list = []
for k in range(num_classes):
    wk = np.dot(Sigma_inv, mu[k])
    wk0 = -0.5 * np.dot(np.dot(mu[k].T, Sigma_inv), mu[k]) + np.log(priors[k])
    wk_list.append(wk)
    wk0_list.append(wk0)

# Convert to numpy arrays for convenience
wk_array = np.array(wk_list)
wk0_array = np.array(wk0_list)

print(wk_array)
print(wk0_array)

# Calculate ak(x)
a = np.array([np.dot(X_test, wk_array[k]) + wk0_array[k] for k in range(num_classes)])

# Softmax function
exp_a = np.exp(a - np.max(a, axis=0))
posterior = exp_a / np.sum(exp_a, axis=0)

# Predicted class
y_pred = np.argmax(posterior, axis=0)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
