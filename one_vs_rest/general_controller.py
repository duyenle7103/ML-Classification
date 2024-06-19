import numpy as np
from obj import InputData
from data_loader import load_iris_data
from data_processor import generate_combinations, kfold_split
from trainer import train_classifier
from predictor import predict, evaluate

PATH = 'input/iris.data'
NUM_SPLIT = 5

def main():
    # Create InputData object
    input_data = InputData()

    # Load data
    input_data.X, input_data.Y, input_data.numtypes = load_iris_data(PATH)

    # Split data into training and testing sets
    for X_train, Y_train, X_test, Y_test in kfold_split(input_data.X, input_data.Y, n_splits=NUM_SPLIT):
        # Initialize weights
        weights = []

        # Run discriminative classifier for each pair
        for i in range(input_data.numtypes):
            # Convert labels to 1 and -1 for current class vs. rest
            Y_train_binary = np.where(Y_train == i, 1, -1)

            # Train classifier and predict
            W = train_classifier(X_train, Y_train_binary)
            weights.append(W)
            
        # Final prediction
        Y_pred = predict(weights, X_test)
        accuracy = evaluate(Y_test, Y_pred)
        input_data.total_accuracy += accuracy

        print(f'Accuracy for this fold: {accuracy * 100:.6f}%')

    print(f'Mean accuracy: {input_data.total_accuracy / NUM_SPLIT * 100:.6f}%')

    