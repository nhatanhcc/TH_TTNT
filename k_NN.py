# --- Phần 1: Chuẩn bị Dữ liệu và Đánh giá K bằng Hình ảnh ---

# Bước 1: Import các thư viện cần thiết
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # datavisualization
from sklearn.datasets import make_blobs # synthetic dataset
from sklearn.neighbors import KNeighborsClassifier # kNN classifier
from sklearn.model_selection import train_test_split # train and test sets 
from sklearn.model_selection import GridSearchCV # Tìm K tối ưu tự động

# Bước 2: Khởi tạo tập dữ liệu
# Tạo 100 mẫu, 2 đặc trưng, 4 cụm (lớp), độ lệch chuẩn 1.
X, y = make_blobs(n_samples=100, n_features=2, centers=4, cluster_std=1, random_state=4)
 
# Bước 3: Xem phân bố của dữ liệu gốc
print("Bước 3: Xem phân bố của dữ liệu gốc (100 mẫu)")
plt.figure(figsize=(9, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', s=50)
plt.title('Phân bố Dữ liệu Gốc (X, y)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Bước 4: Chia tập dữ liệu thành tập huấn luyện (75%) và kiểm tra (25%)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
 
# Xem phân bố của tập kiểm tra (test set)
print("\nBước 4: Phân bố của Tập Kiểm tra (Test Set)")
plt.figure(figsize=(9, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='o', s=40)
plt.title('Phân bố Tập Kiểm tra (Nhãn Thật)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Bước 5: Huấn luyện và đánh giá (nhìn hình) với k=5
knn5 = KNeighborsClassifier(5) # k=5
knn5.fit(X_train, y_train)

y_pred_5 = knn5.predict(X_test)

print("\nBước 5: Dự đoán và Đánh giá với k=5")
plt.figure(figsize=(9, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_5, marker='o', s=40)
plt.title('Dự đoán k-NN với k=5')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
print(f"Độ chính xác (Accuracy) với k=5: {knn5.score(X_test, y_test):.2f}")


# Bước 6: Huấn luyện và đánh giá (nhìn hình) với k=1
knn1 = KNeighborsClassifier(1) # k=1
knn1.fit(X_train, y_train)

y_pred_1 = knn1.predict(X_test)

print("\nBước 6: Dự đoán và Đánh giá với k=1")
plt.figure(figsize=(9, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_1, marker='o', s=40)
plt.title('Dự đoán k-NN với k=1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
print(f"Độ chính xác (Accuracy) với k=1: {knn1.score(X_test, y_test):.2f}")


# --- Phần 2: Tìm K Tối ưu bằng GridSearchCV ---

# Bước 4 (cách 2): Sử dụng GridSearchCV để tìm K tối ưu
print("\n--- Phần 2: Tìm K Tối ưu Tự động bằng GridSearchCV ---")
print("Bước 4 (Cách 2): Tìm giá trị k tối ưu trong khoảng [1, 9]")

# Khởi tạo GridSearchCV: Kiểm tra mô hình KNeighborsClassifier
# với các giá trị 'n_neighbors' từ 1 đến 9, sử dụng Cross-Validation 5 lần (cv=5)
knn_grid = GridSearchCV(estimator=KNeighborsClassifier(), 
                        param_grid={'n_neighbors': np.arange(1, 10)}, 
                        cv=5)

# Huấn luyện trên toàn bộ dữ liệu X, y
knn_grid.fit(X, y)

# In ra tham số k tốt nhất tìm được
print (f"Giá trị k tối ưu (best_params_): {knn_grid.best_params_}")
print(f"Độ chính xác tốt nhất (best_score_): {knn_grid.best_score_:.4f}")


# --- Phần 3: Triển khai Hàm K-NN Tự Xây Dựng ---

# Bước 5 (cách 2): Viết hàm dán nhãn dữ liệu dùng kĩ thuật k-NN dựa vào dữ liệu
# Lưu ý: Hàm này tính khoảng cách Euclidean và trả về nhãn đa số.
def KNN(X_train, X_test, y_train, k):
    num_test = X_test.shape[0] # số lượng dữ liệu test
    num_train = X_train.shape[0] # số lượng dữ liệu train
    
    # Ma trận chứa khoảng cách: (num_test, num_train)
    distances = np.zeros((num_test, num_train))
    
    # Tính khoảng cách Euclidean
    for i in range(num_test):
        # Tính khoảng cách giữa điểm test thứ i với tất cả các điểm train
        distances[i, :] = np.sqrt(np.sum(np.power(X_test[i, :] - X_train, 2), axis=1))
        
    results = []
    
    # Sắp xếp và tìm nhãn đa số
    for i in range(num_test):
        # Kết hợp khoảng cách và nhãn của tập train
        zipped = zip(distances[i, :], y_train)
        # Sắp xếp theo khoảng cách (phần tử đầu tiên của tuple)
        res = sorted(zipped, key=lambda x: x[0]) 
        # Lấy K hàng xóm gần nhất
        results_topk = res[:k]
        
        # Đếm số lượng của mỗi class trong K hàng xóm
        classes = {}
        for _, j in results_topk:
            j = int(j)
            if j not in classes:
                classes[j] = 1
            else:
                classes[j] += 1 
        
        # Trả về class có số lượng nhiều nhất
        # max(classes, key=classes.get) trả về key (class) có value (số lần xuất hiện) lớn nhất
        results.append(max(classes, key=classes.get))
        
    return np.array(results)

# Tạo dữ liệu mới để kiểm tra hàm KNN tự xây dựng
print("\n--- Phần 3: Kiểm tra Hàm K-NN Tự Xây Dựng ---")
# Dữ liệu train/test kích thước lớn hơn
(X_large, y_large) = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=1, random_state=4)
# Điểm dữ liệu mới cần dự đoán nhãn
X_test_new = np.array([(1, 3)])
# Sử dụng k=3
k_val = 3

# Gọi hàm KNN tự xây dựng
results = KNN(X_large, X_test_new, y_large, k_val)
print (f"Dự đoán nhãn cho điểm {X_test_new[0]} với k={k_val}: {results}")

#