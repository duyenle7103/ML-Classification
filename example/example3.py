import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load the IRIS dataset
iris = load_iris()
X = iris.data
y = iris.target

# One-hot encode the class labels
encoder = OneHotEncoder(sparse_output=False)
T = encoder.fit_transform(y.reshape(-1, 1))

# Add a column of ones to X for the bias term
X_ = np.hstack([np.ones((X.shape[0], 1)), X])

# Split the data into training and testing sets
X_train, X_test, T_train, T_test = train_test_split(X_, T, test_size=0.3, random_state=42)

# Calculate the pseudoinverse of X_
X_pseudo_inverse = np.linalg.pinv(X_train)

# Compute the weight matrix W
W = np.dot(X_pseudo_inverse, T_train)

# Define the discriminant function
def discriminant_function(X, W):
    return np.dot(X, W)

# Apply the discriminant function to classify the test data
y_pred = discriminant_function(X_test, W)
y_pred_class = np.argmax(y_pred, axis=1)

# Calculate the accuracy
accuracy = np.mean(y_pred_class == np.argmax(T_test, axis=1))
print(f'Accuracy: {accuracy:.2f}')

# Test the classifier with new data points
new_data = np.array([[1, 6.0, 2.9, 4.5, 1.5],
                     [1, 5.5, 2.4, 3.7, 1.0],
                     [1, 5.0, 3.6, 1.4, 0.2],
                     [1, 5.4, 3.9, 1.7, 0.4],
                     [1, 5.7, 2.5, 5.0, 2.0],
                     [1, 7.0, 3.2, 4.7, 1.4]])

new_predictions = discriminant_function(new_data, W)
new_pred_class = np.argmax(new_predictions, axis=1)
print(f'Predicted classes for new data: {new_pred_class}')
