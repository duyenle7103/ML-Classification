import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import multivariate_normal
from sklearn.naive_bayes import GaussianNB

# Load the dataset
data = load_iris()
X = data.data
y = data.target

# Encode the labels
labels = np.unique(y)
y_encoded = np.array([np.where(labels == label)[0][0] for label in y])

# Function to compute the discriminant function classifier
def discriminant_function(X_train, y_train, X_test):
    means = np.array([X_train[y_train == i].mean(axis=0) for i in np.unique(y_train)])
    cov_matrix = np.cov(X_train, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    weights = np.dot(inv_cov_matrix, means.T).T
    biases = -0.5 * np.sum(means * np.dot(inv_cov_matrix, means.T).T, axis=1) + np.log([np.mean(y_train == i) for i in np.unique(y_train)])
    
    z = np.dot(X_test, weights.T) + biases
    return np.argmax(z, axis=1)

# Function to compute the Bayesian classifier using softmax
def bayesian_classifier(X_train, y_train, X_test):
    means = np.array([X_train[y_train == i].mean(axis=0) for i in np.unique(y_train)])
    cov_matrix = np.cov(X_train, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    weights = np.dot(inv_cov_matrix, means.T).T
    biases = -0.5 * np.sum(means * np.dot(inv_cov_matrix, means.T).T, axis=1) + np.log([np.mean(y_train == i) for i in np.unique(y_train)])
    
    z = np.dot(X_test, weights.T) + biases
    exp_z = np.exp(z)
    softmax_probs = exp_z / exp_z.sum(axis=1, keepdims=True)
    return np.argmax(softmax_probs, axis=1)

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# One-Versus-One (OVO)
ovo_results = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]
    
    classifiers = []
    for (i, j) in [(0, 1), (0, 2), (1, 2)]:
        idx = np.where((y_train == i) | (y_train == j))[0]
        X_train_ij = X_train[idx]
        y_train_ij = y_train[idx]
        clf = discriminant_function(X_train_ij, y_train_ij, X_test)
        classifiers.append(clf)
    
    # Majority vote
    y_pred = np.array(classifiers).T
    y_pred_final = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=y_pred)
    ovo_results.append(accuracy_score(y_test, y_pred_final))

# One-Versus-Rest (OVR)
ovr_results = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]
    
    classifiers = []
    for i in range(3):
        y_train_ovr = np.where(y_train == i, i, -1)
        clf = discriminant_function(X_train, y_train_ovr, X_test)
        classifiers.append(clf)
    
    # Majority vote
    y_pred = np.array(classifiers).T
    y_pred_final = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=y_pred)
    ovr_results.append(accuracy_score(y_test, y_pred_final))

# 3-Class Discriminant Function Classifier
df_results = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]
    y_pred = discriminant_function(X_train, y_train, X_test)
    df_results.append(accuracy_score(y_test, y_pred))

# 3-Class Fisher’s Discriminant Function Classifier
fisher_results = []
lda = LinearDiscriminantAnalysis(n_components=2)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]
    
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    
    y_pred = discriminant_function(X_train_lda, y_train, X_test_lda)
    fisher_results.append(accuracy_score(y_test, y_pred))

# 3-Class Bayesian Classifier
bayesian_results = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]
    y_pred = bayesian_classifier(X_train, y_train, X_test)
    bayesian_results.append(accuracy_score(y_test, y_pred))

# Report results
print(f'OVO Accuracy: {np.mean(ovo_results) * 100:.2f}%')
print(f'OVR Accuracy: {np.mean(ovr_results) * 100:.2f}%')
print(f'Discriminant Function Accuracy: {np.mean(df_results) * 100:.2f}%')
print(f'Fisher’s Discriminant Function Accuracy: {np.mean(fisher_results) * 100:.2f}%')
print(f'Bayesian Classifier Accuracy: {np.mean(bayesian_results) * 100:.2f}%')
