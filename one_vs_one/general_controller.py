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
    input_data.combinations = generate_combinations(input_data.numtypes)

    # Split data into training and testing sets
    for X_train, Y_train, X_test, Y_test in kfold_split(input_data.X, input_data.Y, n_splits=NUM_SPLIT):
        # Initialize matrix to store votes
        votes = np.zeros((X_test.shape[0], input_data.numtypes))

        # Run discriminative classifier for each pair
        for (i, j) in input_data.combinations:
            # Extract data for the current pair and convert labels to 1 and -1
            mask_train = np.isin(Y_train, [i, j])
            mask_test = np.isin(Y_test, [i, j])

            X_train_binary = X_train[mask_train]
            Y_train_binary = Y_train[mask_train]
            Y_train_binary = np.where(Y_train_binary == i, 1, -1)

            X_test_binary = X_test[mask_test]

            # Train classifier and predict
            W = train_classifier(X_train_binary, Y_train_binary)
            Y_pred = predict(W, X_test_binary)

            for idx, prediction in zip(np.where(mask_test)[0], Y_pred):
                if prediction == 1:
                    votes[idx, i] += 1
                else:
                    votes[idx, j] += 1
            
        # Final prediction
        Y_pred_final = np.argmax(votes, axis=1)
        accuracy = evaluate(Y_test, Y_pred_final)
        input_data.total_accuracy += accuracy

        print(f'Accuracy for this fold: {accuracy * 100:.6f}%')

    print(f'Mean accuracy: {input_data.total_accuracy / NUM_SPLIT * 100:.6f}%')