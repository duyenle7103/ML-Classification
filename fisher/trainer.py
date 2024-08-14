import numpy as np

def pseudo_inverse(X):
    return np.linalg.pinv(X)

def cal_SW(X_train, Y_train, numtypes, mean_vectors):
    num_features = X_train.shape[1]
    S_W = np.zeros((num_features, num_features))

    # Compute within-class scatter matrix
    for cl, mv in zip(range(numtypes), mean_vectors):
        class_sc_mat = np.zeros((num_features, num_features))
        for row in X_train[Y_train == cl]:
            row = row.reshape(num_features, 1)
            mv = mv.reshape(num_features, 1)
            class_sc_mat += (row - mv).dot((row - mv).T)
        S_W += class_sc_mat

    return S_W

def cal_SB(X_train, Y_train, mean_vectors):
    num_features = X_train.shape[1]
    overall_mean = np.mean(X_train, axis=0)
    S_B = np.zeros((num_features, num_features))

    # Compute between-class scatter matrix
    for i, mean_vec in enumerate(mean_vectors):
        n = X_train[Y_train == i, :].shape[0]
        mean_vec = mean_vec.reshape(num_features, 1)
        overall_mean = overall_mean.reshape(num_features, 1)
        S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

    return S_B

def cal_project_matrix(S_W, S_B):
    # Compute Fisher's criterion
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

    # Create pairs of eigenvalues and eigenvectors
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

    # Select two eigenvectors corresponding to the two largest eigenvalues
    W = np.hstack((eig_pairs[0][1].reshape(4,1), eig_pairs[1][1].reshape(4,1))).real

    return W

def train_classifier(X_train, X_train_fisher, Y_train, numtypes):
    # Augment data matrix X_train with bias term
    X_train_augmented = np.hstack((np.ones((X_train.shape[0], 1)), X_train_fisher))
    T_train = np.zeros((X_train.shape[0], numtypes))
    for i in range(X_train.shape[0]):
        T_train[i, Y_train[i]] = 1

    # Compute pseudo-inverse of augmented data matrix X_train
    X_train_augmented_pseudo_inverse = pseudo_inverse(X_train_augmented)
    W_ = X_train_augmented_pseudo_inverse.dot(T_train)

    return W_