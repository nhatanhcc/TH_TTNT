import copy
from heapq import heappush, heappop

# ------------------------------
# Hướng di chuyển: Dưới, Trái, Trên, Phải
# ------------------------------
rows = [1, 0, -1, 0]
cols = [0, -1, 0, 1]

class PriorityQueue:
    def __init__(self):
        self.heap = []

    def push(self, key):
        heappush(self.heap, key)

    def pop(self):
        return heappop(self.heap)

    def empty(self):
        return len(self.heap) == 0

class Node:
    def __init__(self, parent, matrix, blank_pos, cost, level):
        self.parent = parent
        self.matrix = matrix
        self.blank_pos = blank_pos
        self.cost = cost
        self.level = level

    # So sánh để Priority Queue sắp xếp (Cost thấp + Level thấp được ưu tiên)
    def __lt__(self, other):
        return (self.cost + self.level) < (other.cost + other.level)

# ------------------------------
# CẢI TIẾN 1: Heuristic Manhattan Distance
# Hiệu quả hơn đếm ô sai vị trí cho ma trận lớn
# ------------------------------
def calculate_cost(matrix, goal_dict, n):
    cost = 0
    for r in range(n):
        for c in range(n):
            val = matrix[r][c]
            if val != 0: # Không tính ô trống
                target_r, target_c = goal_dict[val]
                cost += abs(r - target_r) + abs(c - target_c)
    return cost

def new_node(matrix, blank_pos, new_blank_pos, level, parent, goal_dict, n):
    # Sao chép ma trận (dùng list comprehension nhanh hơn deepcopy một chút cho list lồng)
    new_matrix = [row[:] for row in matrix]
    
    x1, y1 = blank_pos
    x2, y2 = new_blank_pos
    
    # Hoán đổi vị trí
    new_matrix[x1][y1], new_matrix[x2][y2] = new_matrix[x2][y2], new_matrix[x1][y1]
    
    cost = calculate_cost(new_matrix, goal_dict, n)
    return Node(parent, new_matrix, new_blank_pos, cost, level)

def is_safe(x, y, n):
    return 0 <= x < n and 0 <= y < n

def print_matrix(matrix):
    for row in matrix:
        # In căn chỉnh đẹp (padding 3 ký tự)
        print(" ".join(f"{x:2d}" for x in row))
    print()

def print_path(node):
    if node is None:
        return
    print_path(node.parent)
    print(f"Bước {node.level}:")
    print_matrix(node.matrix)

# ------------------------------
# CẢI TIẾN 2: Kiểm tra tính giải được (Solvability)
# Tránh treo máy nếu người dùng nhập đề bài vô nghiệm
# ------------------------------
def get_inversions(arr):
    inv_count = 0
    flat_arr = [x for row in arr for x in row if x != 0]
    for i in range(len(flat_arr)):
        for j in range(i + 1, len(flat_arr)):
            if flat_arr[i] > flat_arr[j]:
                inv_count += 1
    return inv_count

def is_solvable(matrix, n):
    inv_count = get_inversions(matrix)
    
    # Tìm hàng của ô trống (tính từ dưới lên)
    blank_row = 0
    for i in range(n - 1, -1, -1):
        for j in range(n):
            if matrix[i][j] == 0:
                blank_row = n - i
                break
    
    if n % 2 == 1: # Kích thước lẻ (3x3, 5x5...)
        return inv_count % 2 == 0
    else: # Kích thước chẵn (4x4, 6x6...)
        if blank_row % 2 == 1: # Ô trống ở hàng lẻ từ dưới lên
            return inv_count % 2 == 0
        else: # Ô trống ở hàng chẵn từ dưới lên
            return inv_count % 2 != 0

# ------------------------------
# Thuật toán chính
# ------------------------------
def solve_puzzle(initial, blank_pos, goal, n):
    # Tạo dictionary cho goal để tra cứu nhanh toạ độ khi tính Manhattan
    goal_dict = {}
    for r in range(n):
        for c in range(n):
            goal_dict[goal[r][c]] = (r, c)

    pq = PriorityQueue()
    cost = calculate_cost(initial, goal_dict, n)
    root = Node(None, initial, blank_pos, cost, 0)
    pq.push(root)
    
    # CẢI TIẾN 3: Tập visited để tránh lặp lại trạng thái cũ (Tối ưu tốc độ)
    visited = set()

    while not pq.empty():
        current = pq.pop()

        # Chuyển ma trận thành tuple để lưu vào set (hashable)
        current_state_tuple = tuple(tuple(row) for row in current.matrix)
        
        if current.cost == 0:
            print(f"Đã tìm thấy lời giải sau {current.level} bước di chuyển!\n")
            print_path(current)
            return

        # Nếu trạng thái này đã duyệt rồi thì bỏ qua
        if current_state_tuple in visited:
            continue
        visited.add(current_state_tuple)

        # Sinh trạng thái con
        for i in range(4):
            new_x = current.blank_pos[0] + rows[i]
            new_y = current.blank_pos[1] + cols[i]

            if is_safe(new_x, new_y, n):
                child = new_node(
                    current.matrix,
                    current.blank_pos,
                    [new_x, new_y],
                    current.level + 1,
                    current,
                    goal_dict,
                    n
                )
                # Chỉ push vào hàng đợi nếu chưa từng duyệt qua
                child_tuple = tuple(tuple(row) for row in child.matrix)
                if child_tuple not in visited:
                    pq.push(child)

    print("Không tìm thấy lời giải (Dù đã kiểm tra Solvable - Lỗi lạ).")

# ------------------------------
# CHẠY CHƯƠNG TRÌNH
# ------------------------------
if __name__ == "__main__":
    try:
        n = int(input("Nhập kích thước cạnh ma trận N (VD: 3, 4, 5...): "))
        
        print(f"\nNhập ma trận {n}x{n} (các số cách nhau bởi dấu cách, dùng 0 cho ô trống):")
        initial = []
        for i in range(n):
            row = list(map(int, input(f"Hàng {i + 1}: ").split()))
            if len(row) != n:
                print(f"Lỗi: Bạn phải nhập đủ {n} số.")
                exit()
            initial.append(row)

        # CẢI TIẾN 4: Tự động tạo ma trận đích chuẩn
        # VD 3x3: [[1,2,3], [4,5,6], [7,8,0]]
        goal = []
        count = 1
        for i in range(n):
            row = []
            for j in range(n):
                row.append(count)
                count += 1
            goal.append(row)
        goal[n-1][n-1] = 0 # Ô cuối cùng là 0
        
        print("\nMa trận đích mong muốn:")
        print_matrix(goal)

        # Tìm vị trí ô trống
        blank_pos = None
        for i in range(n):
            for j in range(n):
                if initial[i][j] == 0:
                    blank_pos = [i, j]

        # Kiểm tra tính hợp lệ trước khi giải
        if not is_solvable(initial, n):
            print("\nCẢNH BÁO: Ma trận này KHÔNG THỂ GIẢI ĐƯỢC (Unsolvable)!")
            print("Lý do: Số cặp nghịch thế không thỏa mãn điều kiện toán học của trò chơi.")
        else:
            print("\nĐang giải bài toán (có thể mất chút thời gian với N lớn)...")
            solve_puzzle(initial, blank_pos, goal, n)

    except ValueError:
        print("Lỗi: Vui lòng nhập số nguyên hợp lệ.")