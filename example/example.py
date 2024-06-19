import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# Tải tập dữ liệu IRIS
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Các kết hợp từng cặp một
combinations = [(0, 1), (0, 2), (1, 2)]

# Hàm tính toán giả nghịch đảo
def pseudo_inverse(X):
    return np.linalg.pinv(X)

# Hàm huấn luyện bộ phân loại nhị phân
def train_classifier(X, y):
    X_tilde = np.hstack((np.ones((X.shape[0], 1)), X))  # Thêm bias term
    X_pseudo_inv = pseudo_inverse(X_tilde)
    T = np.where(y == 1, 1, -1)  # Chuyển nhãn thành 1 và -1
    W = np.dot(X_pseudo_inv, T)
    return W

# Hàm dự đoán
def predict(W, X):
    X_tilde = np.hstack((np.ones((X.shape[0], 1)), X))  # Thêm bias term
    y_pred = np.dot(X_tilde, W)
    return np.where(y_pred >= 0, 1, -1)

# Lặp qua từng cặp kết hợp
for (i, j) in combinations:
    # Trích xuất dữ liệu cho cặp hiện tại
    mask = np.isin(y, [i, j])
    X_binary = X[mask]
    y_binary = y[mask]
    y_binary = np.where(y_binary == i, 1, -1)  # Chuyển nhãn thành 1 và -1

    # Chia thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.2, random_state=42)

    # Huấn luyện bộ phân loại
    W = train_classifier(X_train, y_train)

    # Dự đoán trên tập kiểm tra
    y_pred = predict(W, X_test)

    # Đánh giá bộ phân loại
    accuracy = np.mean(y_pred == y_test)
    print(f'Độ chính xác cho các lớp {i} vs {j}: {accuracy:.2f}')
