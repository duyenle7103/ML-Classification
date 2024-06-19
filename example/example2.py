import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# Load the IRIS dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, header=None)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# Map class labels to integers
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['class']))}
df['class'] = df['class'].map(class_mapping)

# Split the data into features and labels
X = df.iloc[:, :-1].values
y = df['class'].values

# Add bias term to the feature matrix
X_bias = np.hstack((np.ones((X.shape[0], 1)), X))

# Convert labels to one-versus-the-rest format
def one_vs_rest_labels(y, class_idx):
    return np.where(y == class_idx, 1, 0)

# Define the discriminant function
def discriminant_function(x, W):
    return np.dot(W.T, x)

# Predict function using one-versus-the-rest
def predict(X, weights):
    scores = np.dot(X, np.array(weights).T)
    return np.argmax(scores, axis=1)

# K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []

for train_index, test_index in kf.split(X_bias):
    X_train, X_test = X_bias[train_index], X_bias[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Compute the pseudo-inverse of X_train
    X_pseudo_inverse = np.linalg.pinv(X_train)

    # Initialize a list to store the weights for each classifier
    weights = []

    # Train a classifier for each class
    for class_idx in range(len(class_mapping)):
        T = one_vs_rest_labels(y_train, class_idx)
        W = np.dot(X_pseudo_inverse, T)
        weights.append(W)

    # Make predictions on the test set
    y_pred = predict(X_test, weights)

    # Evaluate accuracy
    accuracy = np.mean(y_pred == y_test)
    accuracies.append(accuracy)

# Calculate and print the average accuracy
average_accuracy = np.mean(accuracies)
print(f'Average Accuracy: {average_accuracy * 100:.2f}%')
