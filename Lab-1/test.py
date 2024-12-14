import unittest
from spareMatrix import (SparseMatrix, sum_matrix, scalar_multiplication,
                         multiplication_matrix, determinant, is_inverse_exists)

class TestSparseMatrix(unittest.TestCase):
    def test_add_element_and_get_element(self):
        matrix = SparseMatrix(3, 3)
        matrix.add_element(1, 1, 5)
        matrix.add_element(2, 3, 10)

        self.assertEqual(matrix.get_element(1, 1), 5)
        self.assertEqual(matrix.get_element(2, 3), 10)
        self.assertEqual(matrix.get_element(3, 3), 0)  # Элемента нет, возвращает 0

    def test_trace(self):
        matrix = SparseMatrix(3, 3)
        matrix.add_element(1, 1, 4)
        matrix.add_element(2, 2, 7)
        matrix.add_element(3, 3, 2)

        self.assertEqual(matrix.trace(), 4 + 7 + 2)

    def test_addition(self):
        matrix1 = SparseMatrix(2, 2)
        matrix1.add_element(1, 1, 2)
        matrix1.add_element(1, 2, 3)

        matrix2 = SparseMatrix(2, 2)
        matrix2.add_element(1, 1, 1)
        matrix2.add_element(2, 2, 4)

        result = sum_matrix(matrix1, matrix2)

        self.assertEqual(result.get_element(1, 1), 3)  # 2 + 1
        self.assertEqual(result.get_element(1, 2), 3)  # 3 + 0
        self.assertEqual(result.get_element(2, 2), 4)  # 0 + 4

    def test_addition_with_zero_matrix(self):
        matrix1 = SparseMatrix(3, 3)
        matrix1.add_element(1, 1, 7)
        matrix1.add_element(2, 2, 8)

        zero_matrix = SparseMatrix(3, 3)  # Матрица без элементов

        result = sum_matrix(matrix1,zero_matrix)

        self.assertEqual(result.get_element(1, 1), 7)  # 7 + 0
        self.assertEqual(result.get_element(2, 2), 8)  # 8 + 0
        self.assertEqual(result.get_element(3, 3), 0)  # 0 + 0

    def test_addition_different_size(self):
        matrix1 = SparseMatrix(2, 2)
        matrix1.add_element(1, 1, 5)

        matrix2 = SparseMatrix(3, 3)
        matrix2.add_element(1, 1, 10)

        with self.assertRaises(ValueError):
            result = sum_matrix(matrix1, matrix2)  # Должно выбросить ошибку

    def test_addition_result_zero(self):
        matrix1 = SparseMatrix(2, 2)
        matrix1.add_element(1, 1, 5)
        matrix1.add_element(2, 2, -3)

        matrix2 = SparseMatrix(2, 2)
        matrix2.add_element(1, 1, -5)
        matrix2.add_element(2, 2, 3)

        result = sum_matrix(matrix1, matrix2)

        self.assertEqual(result.get_element(1, 1), 0)  # 5 + (-5)
        self.assertEqual(result.get_element(2, 2), 0)  # -3 + 3
        self.assertEqual(len(result.elements), 0)  # Никаких ненулевых элементов

    def test_addition_large_sparse(self):
        matrix1 = SparseMatrix(1000, 1000)
        matrix1.add_element(1, 1, 10)
        matrix1.add_element(500, 500, 20)

        matrix2 = SparseMatrix(1000, 1000)
        matrix2.add_element(1, 1, -10)
        matrix2.add_element(999, 999, 30)

        result = sum_matrix(matrix1, matrix2)

        self.assertEqual(result.get_element(1, 1), 0)  # 10 + (-10)
        self.assertEqual(result.get_element(500, 500), 20)  # 20 + 0
        self.assertEqual(result.get_element(999, 999), 30)  # 0 + 30

    def test_scalar_multiplication(self):
        matrix = SparseMatrix(2, 2)
        matrix.add_element(1, 1, 3)
        matrix.add_element(2, 2, 4)

        result = scalar_multiplication(matrix, 2)

        self.assertEqual(result.get_element(1, 1), 6)  # 3 * 2
        self.assertEqual(result.get_element(2, 2), 8)  # 4 * 2

    def test_matrix_multiplication(self):
        matrix1 = SparseMatrix(2, 3)
        matrix1.add_element(1, 1, 1)
        matrix1.add_element(1, 2, 2)
        matrix1.add_element(2, 2, 3)

        matrix2 = SparseMatrix(3, 2)
        matrix2.add_element(1, 1, 4)
        matrix2.add_element(2, 2, 5)
        matrix2.add_element(3, 1, 6)

        result = multiplication_matrix(matrix1, matrix2)

        self.assertEqual(result.get_element(1, 1), 4)  # 4 + 12
        self.assertEqual(result.get_element(1, 2), 2 * 5)  # 10
        self.assertEqual(result.get_element(2, 2), 3 * 5)  # 15

    def test_multiply_matrices_incompatible(self):
        matrix1 = SparseMatrix(2, 2)
        matrix1.add_element(1, 1, 1)
        matrix1.add_element(2, 2, 2)

        matrix2 = SparseMatrix(3, 3)
        matrix2.add_element(1, 1, 3)
        matrix2.add_element(2, 2, 4)

        with self.assertRaises(ValueError):
            result = multiplication_matrix(matrix1, matrix2) # Матрицы несовместимы

    def test_multiply_5x5_matrices(self):
        matrix1 = SparseMatrix(5, 5)
        matrix1.add_element(1, 1, 1)
        matrix1.add_element(1, 2, 2)
        matrix1.add_element(2, 2, 3)
        matrix1.add_element(3, 3, 4)
        matrix1.add_element(4, 4, 5)
        matrix1.add_element(5, 5, 6)

        matrix2 = SparseMatrix(5, 5)
        matrix2.add_element(1, 1, 7)
        matrix2.add_element(2, 1, 8)
        matrix2.add_element(2, 2, 9)
        matrix2.add_element(3, 3, 10)
        matrix2.add_element(4, 4, 11)
        matrix2.add_element(5, 5, 12)

        result = multiplication_matrix(matrix1, matrix2)

        self.assertEqual(result.get_element(1, 1), 1 * 7 + 2 * 8)
        self.assertEqual(result.get_element(1, 2), 2 * 9)
        self.assertEqual(result.get_element(2, 1), 3 * 8)
        self.assertEqual(result.get_element(2, 2), 3 * 9)
        self.assertEqual(result.get_element(3, 3), 4 * 10)
        self.assertEqual(result.get_element(4, 4), 5 * 11)
        self.assertEqual(result.get_element(5, 5), 6 * 12)

    def test_determinant(self):
        matrix = SparseMatrix(2, 2)
        matrix.add_element(1, 1, 3)
        matrix.add_element(1, 2, 8)
        matrix.add_element(2, 1, 4)
        matrix.add_element(2, 2, 6)

        self.assertEqual(determinant(matrix), 3 * 6 - 8 * 4)

    def test_determinant_2x2(self):
        matrix = SparseMatrix(2, 2)
        matrix.add_element(1, 1, 3)
        matrix.add_element(1, 2, 8)
        matrix.add_element(2, 1, 4)
        matrix.add_element(2, 2, 6)

        self.assertEqual(determinant(matrix), -14)

    def test_determinant_3x3(self):
        matrix = SparseMatrix(3, 3)
        matrix.add_element(1, 1, 6)
        matrix.add_element(1, 2, 1)
        matrix.add_element(1, 3, 1)
        matrix.add_element(2, 1, 4)
        matrix.add_element(2, 2, -2)
        matrix.add_element(2, 3, 5)
        matrix.add_element(3, 1, 2)
        matrix.add_element(3, 2, 8)
        matrix.add_element(3, 3, 7)

        self.assertEqual(determinant(matrix), -306)

    def test_determinant_singular_matrix(self):
        matrix = SparseMatrix(3, 3)
        matrix.add_element(1, 1, 1)
        matrix.add_element(1, 2, 2)
        matrix.add_element(1, 3, 3)
        matrix.add_element(2, 1, 4)
        matrix.add_element(2, 2, 5)
        matrix.add_element(2, 3, 6)
        matrix.add_element(3, 1, 7)
        matrix.add_element(3, 2, 8)
        matrix.add_element(3, 3, 9)

        self.assertEqual(determinant(matrix), 0)

    def test_determinant_1x1(self):
        matrix = SparseMatrix(1, 1)
        matrix.add_element(1, 1, 5)

        self.assertEqual(determinant(matrix), 5)

    def test_determinant_empty_matrix(self):
        matrix = SparseMatrix(3, 3)

        self.assertEqual(determinant(matrix), 0)

    def test_is_inverse_exists(self):
        matrix = SparseMatrix(2, 2)
        matrix.add_element(1, 1, 2)
        matrix.add_element(1, 2, 1)
        matrix.add_element(2, 1, 5)
        matrix.add_element(2, 2, 3)

        self.assertTrue(is_inverse_exists(matrix))  # Определитель не равен 0

        singular_matrix = SparseMatrix(2, 2)
        singular_matrix.add_element(1, 1, 1)
        singular_matrix.add_element(1, 2, 2)
        singular_matrix.add_element(2, 1, 2)
        singular_matrix.add_element(2, 2, 4)

        self.assertFalse(is_inverse_exists(singular_matrix))  # Определитель равен 0


if __name__ == '__main__':
    unittest.main()