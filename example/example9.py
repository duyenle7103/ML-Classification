import numpy as np
from sklearn.model_selection import KFold

# Load the IRIS dataset
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target

# Preprocess the data
X = X / 10.0

# Initialize variables
num_folds = 5
kf = KFold(n_splits=num_folds)
accuracies = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Calculate the within-class scatter matrix Sw
    Sw = np.zeros((X_train.shape[1], X_train.shape[1]))
    for i in range(3):
        class_samples = X_train[y_train == i]
        class_mean = np.mean(class_samples, axis=0)
        diff = class_samples - class_mean
        Sw += np.dot(diff.T, diff)

    # Calculate the between-class scatter matrix Sb
    overall_mean = np.mean(X_train, axis=0)
    Sb = np.zeros((X_train.shape[1], X_train.shape[1]))
    for i in range(3):
        class_samples = X_train[y_train == i]
        class_mean = np.mean(class_samples, axis=0)
        diff = class_mean - overall_mean
        Sb += len(class_samples) * np.outer(diff, diff)

    # Calculate the Fisher's linear discriminant
    eigenvalues, eigenvectors = np.linalg.eigh(np.linalg.inv(Sw) @ Sb)
    W = eigenvectors[:, np.argsort(eigenvalues)[::-1][:2]]

    # Project the data onto the discriminant vectors
    X_train_proj = X_train @ W
    X_test_proj = X_test @ W

    # Define a simple classifier function
    def classify(x):
        distances = [np.linalg.norm(x - X_train_proj[i]) for i in range(len(X_train_proj))]
        return np.argmin(distances)

    # Classify the test set
    y_pred = np.array([classify(x) for x in X_test_proj])

    # Evaluate the accuracy
    accuracy = np.mean(y_pred == y_test)
    accuracies.append(accuracy)

# Calculate the average accuracy
avg_accuracy = np.mean(accuracies)
print("Average accuracy:", avg_accuracy)
