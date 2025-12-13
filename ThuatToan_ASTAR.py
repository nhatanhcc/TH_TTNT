class Graph:
    def __init__(self, adjac_lis):
        self.adjac_lis = adjac_lis

    # Lấy danh sách kề của một đỉnh
    def get_neighbors(self, v):
        return self.adjac_lis.get(v, [])

    # Hàm heuristic — có thể sửa theo yêu cầu
    def h(self, n, heuristic_values):
        return heuristic_values.get(n, 1)

    def a_star_algorithm(self, start, stop, heuristic_values):
        open_list = set([start])
        closed_list = set()

        g_cost = {start: 0}
        parent = {start: start}

        while open_list:
            n = None
            # Tìm đỉnh có f = g + h nhỏ nhất
            for v in open_list:
                if n is None or g_cost[v] + self.h(v, heuristic_values) < g_cost[n] + self.h(n, heuristic_values):
                    n = v

            if n is None:
                print('Không tồn tại đường đi!')
                return None

            # Nếu đã tới đích
            if n == stop:
                path = []
                while parent[n] != n:
                    path.append(n)
                    n = parent[n]
                path.append(start)
                path.reverse()
                print("Đường đi tìm được:", " -> ".join(path))
                print("Tổng chi phí:", g_cost[stop])
                return path

            # Xét các đỉnh kề
            for (m, weight) in self.get_neighbors(n):
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parent[m] = n
                    g_cost[m] = g_cost[n] + weight
                else:
                    if g_cost[m] > g_cost[n] + weight:
                        g_cost[m] = g_cost[n] + weight
                        parent[m] = n
                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add(m)

            open_list.remove(n)
            closed_list.add(n)

        print('Không tồn tại đường đi!')
        return None


# ==============================
# Nhập dữ liệu từ bàn phím
# ==============================
if __name__ == "__main__":
    adjac_lis = {}

    print("=== NHẬP ĐỒ THỊ CHO THUẬT TOÁN A* ===")
    num_vertices = int(input("Nhập số đỉnh: "))

    print("\nNhập tên các đỉnh (vd: A B C D):")
    vertices = input().split()

    # Nhập danh sách kề
    print("\nNhập các cạnh (đỉnh_kề và trọng số).")
    print("Ví dụ: nếu A có cạnh đến B trọng số 3 và C trọng số 1 → nhập: B 3 C 1")
    print("Nếu không có cạnh → chỉ nhấn Enter.")

    for v in vertices:
        data = input(f"Đỉnh {v}: ").split()
        neighbors = []
        for i in range(0, len(data), 2):
            if i + 1 < len(data):
                neighbors.append((data[i], int(data[i + 1])))
        adjac_lis[v] = neighbors

    # Nhập giá trị heuristic cho từng đỉnh
    heuristic_values = {}
    print("\nNhập giá trị heuristic (ước lượng khoảng cách còn lại):")
    for v in vertices:
        h_val = int(input(f"h({v}) = "))
        heuristic_values[v] = h_val

    start = input("\nNhập đỉnh bắt đầu: ").strip()
    stop = input("Nhập đỉnh kết thúc: ").strip()

    g = Graph(adjac_lis)
    g.a_star_algorithm(start, stop, heuristic_values)