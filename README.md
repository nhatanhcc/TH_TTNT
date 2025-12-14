# 📊 Phân Tích & Triển Khai Thuật Toán K-Nearest Neighbors (KNN)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Data_Viz-green?style=for-the-badge)

> Dự án thực hành và phân tích chuyên sâu về thuật toán K-Nearest Neighbors (KNN). Bao gồm việc trực quan hóa dữ liệu, so sánh hiệu suất K, tối ưu hóa tham số tự động và tự cài đặt thuật toán từ con số 0.
Tổng Quan Dự Án

Source code này thực hiện 3 nhiệm vụ chính để làm rõ cách hoạt động của KNN:
1.  **Trực quan hóa (Visualization):** So sánh ranh giới phân lớp giữa mô hình phức tạp (K=1) và mô hình tổng quát (K=5) trên dữ liệu giả lập.
2.  **Tối ưu hóa (Optimization):** Sử dụng `GridSearchCV` để tự động tìm giá trị `K` tốt nhất (Hyperparameter Tuning) trong khoảng [1, 9].
3.  **Cài đặt thủ công (Implementation):** Tự viết hàm `KNN` sử dụng khoảng cách Euclidean mà không dùng thư viện `sklearn` để hiểu bản chất toán học.

Công Nghệ Sử Dụng

* **Ngôn ngữ:** Python 3
* **Thư viện chính:**
    * `numpy`, `pandas`: Xử lý ma trận và dữ liệu.
    * `matplotlib`: Vẽ biểu đồ trực quan hóa.
    * `scikit-learn`: Tạo dữ liệu giả lập, mô hình KNN mẫu và công cụ GridSearch.
