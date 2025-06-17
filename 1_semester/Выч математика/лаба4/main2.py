import numpy as np

def qr_decomposition(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    n = A.shape[0]  
    Q = np.eye(n)   
    R = A.copy()    

    for k in range(n - 1):
       
        p = np.zeros(n)
        a_kk = R[k, k] 

        if a_kk != 0:
            norm_a = np.sqrt(np.sum(R[k:, k] ** 2))
            p[k] = a_kk + (1 if a_kk >= 0 else -1) * norm_a
        else:
            p[k] = np.sqrt(2)

        p[k + 1:] = R[k + 1:, k]  

        P = np.eye(n) - 2 * np.outer(p, p) / np.dot(p, p)
        
        Q = Q @ P
        R = P @ R

        print(f"Итерация {k + 1}:")
        print(f"p = \n{p}")
        print(f"P = \n{P}")
        print(f"R после обновления = \n{R}\n")

    return Q, R

def solve_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:

    Q, R = qr_decomposition(A)  
    y = Q.T @ b  
    x = np.linalg.solve(R, y)  
    return x

A1 = np.array([[1, 2, 3],
                [4, 6, 7],
                [8, 9, 0]])
b1 = np.array([6, 12, 24])
x1_exact = np.array([-11.538, 12.923, -2.769])  

Q1, R1 = qr_decomposition(A1)  
print("A1 = \n", A1)
print("Q1 = \n", Q1)
print("R1 = \n", R1)
print("Проверка: Q1 @ R1 = \n", Q1 @ R1)

x1_r = solve_system(A1, b1)  
print("Решение x1_r = \n", x1_r)
print("Разность x1_r - x1 = \n", x1_r - x1_exact)
print("---------------------------------")

# Пример 2
A2 = np.array([[6.03, 13, -17],
                [13, 29.03, -38],
                [-17, -38, 50.03]])
b2 = np.array([2.0909, 4.1509, -5.1191])
x2_exact = np.array([1.03, 1.03, 1.03])  

Q2, R2 = qr_decomposition(A2)  
print("A2 = \n", A2)
print("Q2 = \n", Q2)
print("R2 = \n", R2)
print("Проверка: Q2 @ R2 = \n", Q2 @ R2)

x2_r = solve_system(A2, b2)  
print("Решение x2_r = \n", x2_r)
print("Разность x2_r - x2 = \n", x2_r - x2_exact)
print("---------------------------------")


A3 = np.array([[2, 0, 1],
                [0, 1, -1],
                [1, 1, 1]])
b3 = np.array([3, 0, 3])

x3_r = solve_system(A3, b3) 
print("Решение x3_r = \n", x3_r)