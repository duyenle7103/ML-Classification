import numpy as np
from obj import InputData
from data_loader import load_iris_data
from data_processor import kfold_split
from predictor import classify, evaluate

PATH = 'input/iris.data'
NUM_SPLIT = 5

def main():
    # Create InputData object
    input_data = InputData()

    # Load data
    input_data.X, input_data.Y, input_data.numtypes = load_iris_data(PATH)
    
    # Split data
    for X_train, Y_train, X_test, Y_test in kfold_split(input_data.X, input_data.Y, NUM_SPLIT):
        # Initialize lists for means, covariances and priors
        class_means = []
        class_covariance = []
        priors = []

        # Calculate the mean vector, covariance matrix and prior for each class
        for i in range(input_data.numtypes):
            class_data = X_train[Y_train == i]
            class_means.append(np.mean(class_data, axis=0))
            class_covariance.append(np.cov(class_data, rowvar=False))
            priors.append(np.sum(Y_train == i) / len(Y_train))

        # Calculate the shared covariance matrix
        shared_cov = np.mean(class_covariance, axis=0)
        sigma_inv = np.linalg.inv(shared_cov)

        # Calculate wk and wk0
        W = []
        W0 = []
        for i in range(input_data.numtypes):
            # Calculate w_k
            mu_k = class_means[i]
            w_k = np.dot(sigma_inv, mu_k)

            # Calculate w_k0
            prior_log = np.log(priors[i])
            w_k0 = -0.5 * np.dot(np.dot(mu_k.T, sigma_inv), mu_k) + prior_log

            # Add w_k and w_k0 to the list
            W.append(w_k)
            W0.append(w_k0)

        # Classify the test data and calculate accuracy
        Y_pred = classify(X_test, W, W0, input_data.numtypes)
        accuracy = evaluate(Y_test, Y_pred)
        input_data.total_accuracy += accuracy

        print(f'Accuracy of this fold: {accuracy * 100:.6f}%')
        
    print(f'Mean accuracy: {input_data.total_accuracy / NUM_SPLIT * 100:.6f}%')