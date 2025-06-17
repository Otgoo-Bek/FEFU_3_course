import numpy as np

def cholesky_decomposition(A):
    n = A.shape[0]
    L = np.zeros_like(A)

    for i in range(n):
        for j in range(i + 1):
            if i == j:  # Если на главной диагонали
                L[i, j] = np.sqrt(A[i, i] - np.sum(L[i, :j] ** 2))
            else:
                L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]
    return L

def solve_cholesky(L, b):
    n = L.shape[0]
    
    # Решение системы Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.sum(L[i, :i] * y[:i])
    
    # Решение системы L^T x = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.sum(L[i, i + 1:] * x[i + 1:])) / L[i, i]
    
    return x

def sqrt_method(A, b):
    L = cholesky_decomposition(A)
    return solve_cholesky(L, b)

# Пример 1
A = np.array([
    [1, 3, -2, 0, -2],
    [3, 4, -5, 1, -3],
    [-2, -5, 3, -2, 2],
    [0, 1, -2, 5, 3],
    [-2, -3, 2, 3, 4]
], dtype=float)

b = np.array([0.5, 5.4, 5.0, 7.5, 3.3], dtype=float)

x = sqrt_method(A, b)
print("Решение x* для первой системы:", x)

# Пример 2
A1 = np.array([
    [5.8, 0.3, -0.2],
    [0.3, 4.0, -0.7],
    [-0.2, -0.7, 6.7]
], dtype=float)

b1 = np.array([3.1, -1.7, 1.1], dtype=float)

x1 = sqrt_method(A1, b1)
print("Решение x* для второй системы:", x1)

# Пример 3
A2 = np.array([
    [4.12, 0.42, 1.34, 0.88],
    [0.42, 3.95, 1.87, 0.43],
    [1.34, 1.87, 3.20, 0.31],
    [0.88, 0.43, 0.31, 5.17]
], dtype=float)

b2 = np.array([11.17, 0.115, 9.909, 9.349], dtype=float)

x2 = sqrt_method(A2, b2)
print("Решение x* для третьей системы:", x2)