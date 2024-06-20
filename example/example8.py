import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from scipy.stats import multivariate_normal
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_loader import load_iris_data

X_train, y_train, numtypes = load_iris_data("input/train.txt")
X_test, y_test, num = load_iris_data("input/test.txt")

# Split the dataset into training and testing sets
print(X_train)
print(X_train.shape)
print(y_train)
print(y_train.shape)
print(X_test)
print(X_test.shape)
print(y_test)
print(y_test.shape)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train)
print(X_test)

# Initialize the number of classes
num_classes = len(np.unique(y_train))

kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []

for train_index, test_index in kf.split(X_train):
    X_fold_train, X_fold_test = X_train[train_index], X_train[test_index]
    y_fold_train, y_fold_test = y_train[train_index], y_train[test_index]

    # Calculate mean and shared covariance matrix
    mu = np.array([np.mean(X_fold_train[y_fold_train == label], axis=0) for label in range(num_classes)])
    Sigma = np.mean([np.cov(X_fold_train[y_fold_train == label], rowvar=False) for label in range(num_classes)], axis=0)

    # Calculate inverse of shared covariance matrix
    Sigma_inv = np.linalg.inv(Sigma)

    # Calculate priors
    priors = [np.mean(y_fold_train == label) for label in range(num_classes)]

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

    # Calculate ak(x)
    a = np.array([np.dot(X_fold_test, wk_array[k]) + wk0_array[k] for k in range(num_classes)])

    # Softmax function
    exp_a = np.exp(a - np.max(a, axis=0))
    posterior = exp_a / np.sum(exp_a, axis=0)

    # Predicted class
    y_pred = np.argmax(posterior, axis=0)

    # Calculate accuracy
    accuracy = accuracy_score(y_fold_test, y_pred)
    accuracies.append(accuracy)

# Average accuracy
avg_accuracy = np.mean(accuracies)
print("Average Accuracy:", avg_accuracy)
