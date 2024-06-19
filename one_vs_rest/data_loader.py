import pandas as pd
import numpy as np

def load_iris_data(path):
    # Read data from file
    dataframe = pd.read_csv(path, sep=',', header=None)

    # Extract features and labels
    X = dataframe.iloc[:, :-1].values
    Y_str = dataframe.iloc[:, -1].values

    # Create dictionary mapping from string to integer
    label_to_int = {label: idx for idx, label in enumerate(pd.unique(Y_str))}
    Y = np.array([label_to_int[label] for label in Y_str])
    numtypes = len(label_to_int)

    return X, Y, numtypes