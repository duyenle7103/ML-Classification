import numpy as np
from obj import InputData
from data_loader import load_iris_data
from data_processor import one_hot_encoder, kfold_split
from trainer import train_classifier
from predictor import predict, evaluate

PATH = 'input/iris.data'
NUM_SPLIT = 5

def main():
    # Create InputData object
    input_data = InputData()

    # Load data
    input_data.X, input_data.Y, input_data.numtypes = load_iris_data(PATH)
    input_data.X = input_data.X / 10

    # Split data
    for X_train, Y_train, X_test, Y_test in kfold_split(input_data.X, input_data.Y, n_splits=NUM_SPLIT):
        # One-hot encode labels
        Y_train_one_hot = one_hot_encoder(Y_train, input_data.numtypes)

        # Train classifier and predict
        W = train_classifier(X_train, Y_train_one_hot)
        Y_pred = predict(W, X_test)

        # Evaluate accuracy
        accuracy = evaluate(Y_test, Y_pred)
        input_data.total_accuracy += accuracy

        print(f"Accuracy for this fold: {accuracy * 100:.6f}%")

    print(f"Mean accuracy: {input_data.total_accuracy / NUM_SPLIT * 100:.6f}%")
    