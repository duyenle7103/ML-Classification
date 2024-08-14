import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from numpy.linalg import inv
from data_loader import load_iris_data

# Tải dữ liệu IRIS
X_train, y_train, class_labels = load_iris_data("input/train.txt")
X_test, y_test, _ = load_iris_data("input/test.txt")

# Bước 2: Tính các vector trung bình cho từng lớp trên tập huấn luyện
mean_vectors = []
for cl in range(class_labels):
    mean_vectors.append(np.mean(X_train[y_train == cl], axis=0))

# Bước 3: Tính ma trận scatter trong lớp (within-class scatter matrix)
S_W = np.zeros((4,4))
for cl, mv in zip(range(class_labels), mean_vectors):
    class_sc_mat = np.zeros((4,4))  # Ma trận scatter cho mỗi lớp
    for row in X_train[y_train == cl]:
        row, mv = row.reshape(4,1), mv.reshape(4,1)  # Chuyển thành vector cột
        class_sc_mat += (row-mv).dot((row-mv).T)
    S_W += class_sc_mat

# Bước 4: Tính ma trận scatter giữa các lớp (between-class scatter matrix)
overall_mean = np.mean(X_train, axis=0)
S_B = np.zeros((4,4))
for i, mean_vec in enumerate(mean_vectors):
    n = X_train[y_train == i, :].shape[0]
    mean_vec = mean_vec.reshape(4,1)
    overall_mean = overall_mean.reshape(4,1)
    S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

# Bước 5: Tính tiêu chuẩn Fisher (Fisher’s criterion) và tìm ma trận chiếu tối ưu W
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

# Tạo danh sách các cặp (giá trị riêng, vector riêng)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sắp xếp các cặp theo thứ tự giảm dần của giá trị riêng
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

# Chọn hai vector riêng tương ứng với hai giá trị riêng lớn nhất
W = np.hstack((eig_pairs[0][1].reshape(4,1), eig_pairs[1][1].reshape(4,1))).real

# Bước 6: Biến đổi dữ liệu sử dụng ma trận chiếu W
X_train_fisher = X_train.dot(W)
X_test_fisher = X_test.dot(W)

# Bước 7: Tính các hàm phân biệt của Fisher sử dụng phương trình 4.17
# Mở rộng ma trận dữ liệu X_train với thành phần bias
X_train_augmented = np.hstack((np.ones((X_train.shape[0], 1)), X_train_fisher))  # Thêm thành phần bias
T_train = np.zeros((X_train.shape[0], class_labels))
for i in range(X_train.shape[0]):
    T_train[i, y_train[i]] = 1

# Tính nghịch đảo giả của ma trận X_train mở rộng
X_train_augmented_pseudo_inverse = np.linalg.pinv(X_train_augmented)
W_ = X_train_augmented_pseudo_inverse.dot(T_train)

# Hàm phân biệt
def y(x):
    x_augmented = np.hstack((1, x)).reshape(-1, 1)  # Thêm thành phần bias
    return W_.T.dot(x_augmented)

# Sử dụng bộ phân loại để phân loại các mẫu trong tập kiểm tra
y_pred = []
for sample in X_test_fisher:
    result = y(sample)
    y_pred.append(np.argmax(result))

# Tính toán độ chính xác
accuracy = np.sum(y_pred == y_test) / len(y_test)
print("Độ chính xác trên tập kiểm tra:", accuracy)
