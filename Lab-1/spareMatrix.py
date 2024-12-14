class SparseMatrix:
    def __init__(self, n, m):
        self.n = n  # Количество строк
        self.m = m  # Количество столбцов
        self.elements = []  # Список для хранения ненулевых элементов: (i, j, value)

    def add_element(self, i, j, value):
        # Добавляем ненулевой элемент в разреженную матрицу
        if value != 0:
            self.elements.append((i, j, value))

    def change_element(self, i, j, value):
        # Ищем элемент с индексами (i, j) в матрице
        for idx, (row, col, val) in enumerate(self.elements):
            if row == i and col == j:
                if value != 0:
                    # Обновляем значение, если оно не ноль
                    self.elements[idx] = (i, j, value)
                else:
                    # Удаляем элемент, если значение становится нулевым
                    self.elements.pop(idx)
                return

        # Если элемента нет и значение не ноль, добавляем его
        if value != 0:
            self.elements.append((i, j, value))

    def get_element(self, i, j):
        # Получаем элемент по индексу (i, j)
        for row, col, value in self.elements:
            if row == i and col == j:
                return value
        return 0  # Если элемент не найден, то это ноль

    def trace(self):
        # Метод для подсчета следа матрицы
        trace_sum = 0
        for i, j, value in self.elements:
            if i == j:  # Добавляем к следу только элементы на главной диагонали
                trace_sum += value
        return trace_sum


def sum_matrix(matrix_1, matrix_2):
    # Метод сложения матриц
    if matrix_1.n != matrix_2.n and matrix_1.m != matrix_2.m:
        raise ValueError("Матрицы должны иметь одинаковые размеры для сложения.")

    result = SparseMatrix(matrix_1.n, matrix_1.m)

    for i in range(1, matrix_1.n + 1):
        for j in range(1, matrix_1.m + 1):
            result.add_element(i, j, matrix_1.get_element(i, j) + matrix_2.get_element(i, j))
    return result


def scalar_multiplication(matrix_1, scalar): # умножение матрици на скаляр
    result = SparseMatrix(matrix_1.n, matrix_1.m)
    for i, j, value in matrix_1.elements:
        result.add_element(i, j, value * scalar)
    return result


def multiplication_matrix(matrix_1, matrix_2): # умножение матриц
    if matrix_1.m != matrix_2.n:
        raise ValueError(
            "Количество столбцов первой матрицы должно быть равно количеству строк второй матрицы.")

    result = SparseMatrix(matrix_1.n, matrix_2.m)

    # Алгоритм умножения матриц (разреженных)
    for i, j, value_1 in matrix_1.elements:
        for k, l, value_2 in matrix_2.elements:
            if j == k:  # Столбец первой матрицы совпадает с строкой второй
                result.change_element(i, l, result.get_element(i,l) + value_1 * value_2)
    return result


def determinant(matrix):
    # Для вычисления определителя нужно преобразовать разреженную матрицу в обычную
    no_spare_matrix = [[matrix.get_element(i + 1, j + 1) for j in range(matrix.m)] for i in range(matrix.n)]
    return _determinant(no_spare_matrix)

def _determinant(matrix):
    # Рекурсивное вычисление определителя
    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for c in range(len(matrix)):
        # Создаем подматрицу, удаляя первую строку и текущий столбец
        submatrix = [row[:c] + row[c + 1:] for row in matrix[1:]]
        # Рекурсивно вычисляем детерминант подматрицы и суммируем с учетом знака
        det += ((-1) ** c) * matrix[0][c] * _determinant(submatrix)
    return det

def is_inverse_exists(matrix):
    # Проверяем, существует ли обратная матрица
    return determinant(matrix) != 0


# Ввод данных для 1 и 2 задачи
n, m = map(int, input("Введите размер матрицы (N M): ").split())
matrix1 = SparseMatrix(n, m)

print("Введите матрицу 1:")
for i in range(1, n + 1):
    row = list(map(float, input().split()))
    for j in range(1, m + 1):
        matrix1.add_element(i, j, row[j - 1])

matrix2 = SparseMatrix(n, m)
print("Введите матрицу 2:")
for i in range(1, n + 1):
    row = list(map(float, input().split()))
    for j in range(1, m + 1):
        matrix2.add_element(i, j, row[j - 1])

# Сложение матриц
try:
    sum_matrix = sum_matrix(matrix1, matrix2)
    print("Сумма матриц:")
    for i in range(1, sum_matrix.n + 1):
        for j in range(1, sum_matrix.m + 1):
            print(sum_matrix.get_element(i, j), end=" ")
        print()
except ValueError as e:
    print(e)

# Умножение матрицы на скаляр
scalar = float(input("Введите скаляр для умножения на первую матрицу: "))

scaled_matrix = scalar_multiplication(matrix1, scalar)
print(f"Матрица после умножения на {scalar}:")
for i in range(1, scaled_matrix.n + 1):
    for j in range(1, scaled_matrix.m + 1):
        print(scaled_matrix.get_element(i, j), end=" ")
    print()

# Умножение матриц
try:
    product_matrix = multiplication_matrix(matrix1, matrix2)
    print("Умножения матриц:")
    for i in range(1, product_matrix.n + 1):
        for j in range(1, product_matrix.m + 1):
            print(product_matrix.get_element(i, j), end=" ")
        print()
except ValueError as e:
    print(e)

# Подсчет следа
print(f"След матрицы 1: {matrix1.trace()}")


# Ввод данных для 3 задачи
print("Ввод для задания номер 3")
n = int(input("Введите размер матрицы (N): "))
matrix = SparseMatrix(n, n)

for i in range(1, n + 1):
    row = list(map(float, input().split()))
    for j in range(1, m + 1):
        matrix.add_element(i, j, row[j - 1])

# Вывод результата
det = determinant(matrix)
print(f"Определитель: {det}")

if is_inverse_exists(matrix):
    print("да")
else:
    print("нет")

