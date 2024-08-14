import numpy as np
from obj import InputData
from data_loader import load_iris_data
from data_processor import kfold_split
from trainer import cal_SW, cal_SB, cal_project_matrix, train_classifier
from predictor import predict, evaluate

PATH = 'input/iris.data'
NUM_SPLIT = 5

def main():
    # Create InputData object
    input_data = InputData()

    # Load data
    input_data.X, input_data.Y, input_data.numtypes = load_iris_data(PATH)
    input_data.X = input_data.X / 10

    # Split data into training and testing sets
    for X_train, Y_train, X_test, Y_test in kfold_split(input_data.X, input_data.Y, n_splits=NUM_SPLIT):
        # Initialize mean vectors
        mean_vectors = []
        for i in range(input_data.numtypes):
            class_data = X_train[Y_train == i]
            mean_vectors.append(np.mean(class_data, axis=0))
        
        # Initialize scatter matrices
        S_W = cal_SW(X_train, Y_train, input_data.numtypes, mean_vectors)
        S_B = cal_SB(X_train, Y_train, mean_vectors)

        # Calculate project matrix
        W = cal_project_matrix(S_W, S_B)

        # Transform data
        X_train_fisher = X_train.dot(W)
        X_test_fisher = X_test.dot(W)

        # Train model
        W_ = train_classifier(X_train, X_train_fisher, Y_train, input_data.numtypes)

        # Predict
        Y_pred = []
        for i in X_test_fisher:
            result = predict(i, W_)
            Y_pred.append(np.argmax(result))

        # Evaluate
        accuracy = evaluate(Y_test, Y_pred)
        input_data.total_accuracy += accuracy

        print(f'Accuracy for this fold: {accuracy * 100:.6f}%')
    
    print(f'Mean accuracy: {input_data.total_accuracy / NUM_SPLIT * 100:.6f}%')
