import numpy as np

class LU:
    def __init__(self, A):
        self.A = A
        self.L = None
        self.U = None

    def decompose(self):
       
        n = self.A.shape[0]
        self.L = np.eye(n)  
        self.U = np.zeros_like(self.A)  

        for i in range(n):
            for j in range(i, n):
                self.U[i, j] = self.A[i, j] - sum(self.L[i, k] * self.U[k, j] for k in range(i))

            for j in range(i + 1, n):
                self.L[j, i] = (self.A[j, i] - sum(self.L[j, k] * self.U[k, i] for k in range(i))) / self.U[i, i]

    def forward_substitution(self, b):
       
        n = self.L.shape[0]
        y = np.zeros(n)
        for i in range(n):
            y[i] = (b[i] - sum(self.L[i, j] * y[j] for j in range(i))) / self.L[i, i]
        return y

    def backward_substitution(self, y):
        
        n = self.U.shape[0]
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (y[i] - sum(self.U[i, j] * x[j] for j in range(i + 1, n))) / self.U[i, i]
        return x

    def solve(self, b):
       
        self.decompose()
        y = self.forward_substitution(b)
        x = self.backward_substitution(y)
        return self.L, self.U, x

    def print_results(self, name, x):
        
        print(f"{name}:")
        print("Матрица L:")
        print(self.L)
        print("\nМатрица U:")
        print(self.U)
        print("\nВектор решений x:")
        print(x)
        print("\n" + "=" * 30 + "\n")


# Определение систем уравнений
A_main = np.array([[5, 2, 3],
                   [1, 6, 1],
                   [3, -4, -2]], dtype=float)

b_main = np.array([3, 5, 8], dtype=float)

A_I = np.array([[13.14, -2.12, 1.17, 0],
                 [-2.12, 6.3, -2.45, 0],
                 [1.17, -2.45, 4.6, 0],
                 [0, 0, 0, 1]], dtype=float)

b_I = np.array([1.27, 2.13, 3.14, 0], dtype=float)

A_II = np.array([[4.31, 0.26, 0.61, 0.27],
                  [0.26, 2.32, 0.18, 0.34],
                  [0.61, 0.18, 3.20, 0.31],
                  [0.27, 0.34, 0.31, 5.17]], dtype=float)

b_II = np.array([1.02, 1.00, 1.34, 1.27], dtype=float)


# Решение систем

# Решение основной системы
lu_main = LU(A_main)
L_main, U_main, x_main = lu_main.solve(b_main)
lu_main.print_results("Основная система", x_main)

# Решение системы I
lu_I = LU(A_I)
L_I, U_I, x_I = lu_I.solve(b_I)
lu_I.print_results("Система I", x_I)

# Решение системы II
lu_II = LU(A_II)
L_II, U_II, x_II = lu_II.solve(b_II)
lu_II.print_results("Система II", x_II)