# Bước 1: Import các thư viện cần thiết
import numpy as np # Thư viện tính toán toán học
import matplotlib.pyplot as plt # Visualize data sử dụng đồ thị
from scipy.spatial.distance import cdist # Hỗ trợ tính khoảng cách

# --- Dữ liệu đầu vào ---
# Bước 2: Khởi tạo 500 điểm dữ liệu xung quanh 3 tâm cụm (2, 2), (9, 2) và (4,9).
means = [[2, 2], [9, 2], [4, 9]]
cov = [[2, 0], [0, 2]] # Ma trận hiệp phương sai
n_samples = 500 # Số lượng mẫu cho mỗi cụm
n_cluster = 3 # Số lượng cụm

# Tạo dữ liệu ngẫu nhiên theo phân phối đa biến (Multivariate Normal Distribution)
X0 = np.random.multivariate_normal(means[0], cov, n_samples)
X1 = np.random.multivariate_normal(means[1], cov, n_samples)
X2 = np.random.multivariate_normal(means[2], cov, n_samples)
X = np.concatenate((X0, X1, X2), axis = 0) # Gộp dữ liệu lại

# Bước 3: Xem phân bố của dữ liệu mà chúng ta vừa tạo ra (Visualize data)
plt.figure(figsize=(6, 6))
plt.xlabel('x')
plt.ylabel('y')
plt.title('Original Data Distribution')
plt.plot(X[:, 0], X[:, 1], 'bo', markersize=4)
plt.show()

# --- Các Hàm của K-Means ---

# Bước 4: Viết hàm khởi tạo n_cluster=3 tâm cụm.
def kmeans_init_centers(X, n_cluster):
  # Chọn ngẫu nhiên n_cluster điểm từ tập dữ liệu X làm tâm cụm ban đầu
  return X[np.random.choice(X.shape[0], n_cluster, replace=False)]

# Bước 5: Hàm xác định tâm cụm (E-step: Gán nhãn)
def kmeans_predict_labels(X, centers):
  # Tính ma trận khoảng cách giữa mỗi điểm dữ liệu và mỗi tâm cụm
  D = cdist(X, centers)
  # Trả về chỉ mục (index) của tâm cụm gần nhất cho mỗi điểm
  return np.argmin(D, axis = 1)

# Bước 6: Hàm cập nhật lại vị trí của các tâm cụm (M-step: Cập nhật tâm)
def kmeans_update_centers(X, labels, n_cluster):
  centers = np.zeros((n_cluster, X.shape[1]))
  for k in range(n_cluster):
    # Lấy tất cả các điểm được gán cho cụm thứ k
    Xk = X[labels == k, :]
    # Tính trung bình (mean) của các điểm để cập nhật vị trí tâm cụm
    centers[k,:] = np.mean(Xk, axis = 0)
  return centers

# Bước 7: Hàm Kiểm tra tính hội tụ.
def kmeans_has_converged(centers, new_centers):
  # Trả về True nếu hai tập tâm cụm là giống nhau (đã hội tụ)
  # Sử dụng set và tuple để so sánh mảng 2 chiều
  return (set([tuple(a) for a in centers]) == 
      set([tuple(a) for a in new_centers]))

# Bước 8: Hàm vẽ lên đồ thị để quan sát kết quả
def kmeans_visualize(X, centers, labels, n_cluster, title):
  plt.figure(figsize=(6, 6))
  plt.xlabel('x')  # label trục x
  plt.ylabel('y')  # label trục y
  plt.title(title)  # title của đồ thị
  # Danh sách màu hỗ trợ cho k <= 4 (khi k > 4 cần thêm màu)
  plt_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'] 

  for i in range(n_cluster):
    # Dữ liệu của cụm i (dùng marker '^')
    data = X[labels == i]  
    plt.plot(data[:, 0], data[:, 1], plt_colors[i] + '^', markersize=4,
             label='cluster_' + str(i))  
    # Tâm cụm i (dùng marker 'o' lớn hơn, màu khác biệt)
    plt.plot(centers[i][0], centers[i][1], plt_colors[i + 4] + 'o', markersize=10,
             label='center_' + str(i))  
  plt.legend()  # Hiện bảng chú thích
  plt.show()

# Bước 9: Toàn bộ thuật toán k-means
def kmeans(init_centes, init_labels, X, n_cluster):
  centers = init_centes
  labels = init_labels
  times = 0 # Biến đếm số lần lặp

  while True:
    # 1. Gán nhãn mới (E-step)
    labels = kmeans_predict_labels(X, centers)
    kmeans_visualize(X, centers, labels, n_cluster, 
                     'Assigned label for data at time = ' + str(times + 1))
    
    # 2. Cập nhật tâm cụm mới (M-step)
    new_centers = kmeans_update_centers(X, labels, n_cluster)
    
    # 3. Kiểm tra hội tụ
    if kmeans_has_converged(centers, new_centers):
      break
    
    # Nếu chưa hội tụ, tiếp tục lặp
    centers = new_centers
    kmeans_visualize(X, centers, labels, n_cluster, 
                     'Update center possition at time = ' + str(times + 1))
    times += 1
    
  return (centers, labels, times)

# --- Thực thi thuật toán ---
# Bước 10: gọi hàm kmeans phía trên để thực thi.
init_centers = kmeans_init_centers(X, n_cluster)
print('Tọa độ khởi tạo ban đầu của các tâm cụm:\n', init_centers)

# Gán nhãn ban đầu (ví dụ: gán tất cả vào cluster 0)
init_labels = np.zeros(X.shape[0]) 

# Visualize trạng thái khởi tạo
kmeans_visualize(X, init_centers, init_labels, n_cluster,
                 'Init centers in the first run. Assigned all data as cluster 0')

# Chạy thuật toán K-Means
centers, labels, times = kmeans(init_centers, init_labels, X, n_cluster)

print('--- KẾT QUẢ CUỐI CÙNG ---')
print('Done! Kmeans has converged after', times, 'times')
print('Tọa độ tâm cụm cuối cùng:\n', centers)
