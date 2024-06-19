import numpy as np
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from itertools import combinations
from collections import Counter

# Load dữ liệu Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Sử dụng StratifiedKFold để giữ cân bằng phân phối nhãn trong mỗi fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    unique_labels = np.unique(y_train)
    classifiers = {}
    
    for (i, j) in combinations(unique_labels, 2):
        mask = (y_train == i) | (y_train == j)
        X_train_ij = X_train[mask]
        y_train_ij = y_train[mask]
        
        class_weights = {i: 1.0, j: 1.0}
        class_counts = Counter(y_train_ij)
        total_samples = len(y_train_ij)
        for label in class_weights:
            class_weights[label] = total_samples / (len(unique_labels) * class_counts[label])
        
        clf = SVC(class_weight=class_weights)
        clf.fit(X_train_ij, y_train_ij)
        classifiers[(i, j)] = clf
    
    votes = np.zeros((X_test.shape[0], len(unique_labels)))
    
    for (i, j), clf in classifiers.items():
        mask = (y_test == i) | (y_test == j)
        X_test_ij = X_test[mask]
        if X_test_ij.shape[0] > 0:
            y_pred = clf.predict(X_test_ij)
            for index, label in zip(np.where(mask)[0], y_pred):
                votes[index, label] += 1
    
    y_pred_final = np.argmax(votes, axis=1)
    accuracy = accuracy_score(y_test, y_pred_final)
    accuracies.append(accuracy)

print("Mean Accuracy: ", np.mean(accuracies))
print("Standard Deviation: ", np.std(accuracies))
