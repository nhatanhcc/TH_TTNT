

Thuật toán KNN (K-Nearest Neighbors)

1.1 Khái niệm

KNN (K-Nearest Neighbors)** là một thuật toán **học có giám sát (Supervised Learning)** dùng cho:

* Phân lớp (Classification)
* Hồi quy (Regression)

Trong đồ án này, KNN được sử dụng cho **bài toán phân lớp dữ liệu 2 chiều**.

Ý tưởng chính:

> Một điểm dữ liệu mới sẽ được gán nhãn dựa trên **K điểm dữ liệu gần nó nhất** trong tập huấn luyện.

---
1.2 Nguyên lý hoạt động

Thuật toán KNN **không xây dựng mô hình học trước**, mà:

* Lưu toàn bộ dữ liệu huấn luyện
* Khi cần dự đoán → mới tính toán

Quy trình:

1. Chọn số láng giềng K
2. Tính khoảng cách từ điểm cần phân loại đến tất cả điểm huấn luyện
3. Chọn K điểm gần nhất
4. Gán nhãn theo **đa số phiếu bầu**

---

### 1.3 Khoảng cách sử dụng

Trong chương trình, thư viện `scikit-learn` mặc định sử dụng **khoảng cách Euclidean**:

[
d(x, y) = \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2}
]

---

1.4 Các bước KNN trong chương trình
 Bước 1: Sinh dữ liệu

```python
X, y = make_blobs(n_samples, centers, cluster_std)
```

* `X`: tọa độ các điểm dữ liệu
* `y`: nhãn lớp tương ứng
* Dữ liệu có nhiễu để mô phỏng dữ liệu thực tế

---

 Bước 2: Chia tập Train/Test

```python
train_test_split(X, y, test_size=0.2)
```

* 80% huấn luyện
* 20% kiểm tra

---
 Bước 3: Huấn luyện KNN

```python
knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(X_train, y_train)
```

* Không huấn luyện theo nghĩa truyền thống
* Chỉ lưu dữ liệu huấn luyện

---

 Bước 4: Dự đoán & đánh giá

```python
y_pred = knn.predict(X_test)
accuracy = knn.score(X_test, y_test)
```

* So sánh nhãn dự đoán và nhãn thực
* Tính độ chính xác

---
1.5 Trực quan hóa KNN trong đồ án

* Tập **Train**: vẽ mờ
* Tập **Test**: vẽ đậm, viền đỏ
* Màu sắc biểu diễn nhãn lớp
Giúp người dùng **quan sát trực quan kết quả phân lớp**

---
1.6 Ưu – Nhược điểm KNN

**Ưu điểm**

* Dễ hiểu, dễ cài đặt
* Không cần huấn luyện phức tạp
* Phù hợp dữ liệu nhỏ

**Nhược điểm**

* Tốn thời gian khi dữ liệu lớn
* Nhạy cảm với giá trị K
* Phụ thuộc mạnh vào khoảng cách

---

 Thuật toán K-Means

---
2.1 Khái niệm

**K-Means** là thuật toán **học không giám sát (Unsupervised Learning)** dùng cho bài toán **phân cụm dữ liệu**.

Mục tiêu:

> Chia tập dữ liệu thành **K cụm**, sao cho các điểm trong cùng cụm **gần nhau nhất**.

---
 2.2 Nguyên lý hoạt động

K-Means hoạt động theo vòng lặp gồm **2 bước chính**:

1. **Gán nhãn (Assignment – E-step)**
2. **Cập nhật tâm cụm (Update – M-step)**

Thuật toán dừng khi **tâm cụm không còn thay đổi**.

---
2.3 Khoảng cách sử dụng

Trong đồ án, khoảng cách Euclidean được tính thủ công bằng:

```python
cdist(X, centers)
```

Công thức:
[
d(x, c) = \sqrt{(x_1 - c_1)^2 + (x_2 - c_2)^2}
]

---

### 2.4 Các bước K-Means trong chương trình

---

 Bước 1: Sinh dữ liệu

```python
X, _ = make_blobs(n_samples, centers)
```

* Dữ liệu **không có nhãn**
* Số cụm thực dùng để so sánh

---

 Bước 2: Khởi tạo tâm cụm

```python
centers = random chọn K điểm
```

* Chọn ngẫu nhiên từ dữ liệu ban đầu

---

 Bước 3: Gán nhãn (Assignment)

```python
labels = argmin(distance(X, centers))
```

* Mỗi điểm được gán vào cụm có tâm gần nhất

---

 Bước 4: Cập nhật tâm (Update)

```python
center_i = mean(các điểm thuộc cụm i)
```

---

 Bước 5: Kiểm tra hội tụ

```python
np.allclose(old_centers, new_centers)
```

* Nếu tâm không đổi → dừng thuật toán

---
2.5 Trực quan hóa K-Means

* Mỗi cụm có **màu riêng**
* Tâm cụm hiển thị bằng dấu **X lớn**
* Thuật toán chạy **từng bước theo thời gian**
**

---

### 2.6 Ưu – Nhược điểm K-Means

**Ưu điểm**

* Nhanh, dễ cài đặt
* Hiệu quả với dữ liệu lớn
* Trực quan

**Nhược điểm**

* Phụ thuộc khởi tạo tâm
* Cần chọn trước K
* Không phù hợp dữ liệu phi tuyến

---

 So sánh KNN và K-Means

| Tiêu chí  | KNN                  | K-Means           |
| --------- | -------------------- | ----------------- |
| Loại học  | Có giám sát          | Không giám sát    |
| Mục tiêu  | Phân lớp             | Phân cụm          |
| Cần nhãn  | Có                   | Không             |
| Giá trị K | Số láng giềng        | Số cụm            |
| Ứng dụng  | Nhận dạng, phân loại | Phân nhóm dữ liệu |

---

 Kết luận

Chương trình đã:

* Mô phỏng thành công **KNN và K-Means**
* Trực quan hóa quá trình học
* Giúp người học hiểu rõ bản chất thuật toán




