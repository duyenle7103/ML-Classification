import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder

# Load the IRIS dataset
iris = datasets.load_iris()
X = iris.data  # Feature matrix
y = iris.target  # Target vector

print(X)
print(y)

# One-hot encode the target vector
one_hot_encoder = OneHotEncoder(sparse_output=False)

print(one_hot_encoder)

T = one_hot_encoder.fit_transform(y.reshape(-1, 1))

print(T)

# Add a bias term (column of ones) to the feature matrix
X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])

print(X_with_bias)

# Compute the pseudo-inverse of X
X_pseudo_inverse = np.linalg.pinv(X_with_bias)

# Calculate the weight matrix W
W = np.dot(X_pseudo_inverse, T)

# Discriminant function
def discriminant_function(X):
    # Add a bias term to the input data
    X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])

    print(X_with_bias)

    return np.dot(X_with_bias, W)

# Classify the data
y_pred = discriminant_function(X)

print(y_pred)

y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate accuracy
accuracy = np.mean(y_pred_classes == y)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Display the results
df_results = pd.DataFrame({'Actual': y, 'Predicted': y_pred_classes})
print(df_results)
