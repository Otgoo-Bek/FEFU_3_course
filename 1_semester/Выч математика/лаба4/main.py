import numpy as np

A1 = np.array([[6.03, 13, -17],
                [13, 29.03, -38],
                [-17, -38, 50.03]])
b1 = np.array([2.0909, 4.1509, -5.1191])
x_exact1 = np.array([1.03, 1.03, 1.03])  


Q1, R1 = np.linalg.qr(A1)
y1 = np.dot(Q1.T, b1)
x_computed1 = np.linalg.solve(R1, y1)


error1 = np.linalg.norm(x_computed1 - x_exact1)

print("Результаты для первой системы:")
print("Вычисленное решение x:", x_computed1)
print("Точное решение x*:", x_exact1)
print("Ошибка:", error1)

A2 = np.array([[2, 0, 1],
                [0, 2, 1],
                [1, 1, 3]])
b2 = np.array([3, 0, 3])


Q2, R2 = np.linalg.qr(A2)
y2 = np.dot(Q2.T, b2)
x_computed2 = np.linalg.solve(R2, y2)

print("\nРезультаты для второй системы:")
print("Вычисленное решение x:", x_computed2)


print("\nИтоги:")
print("Для первой системы Ax = b:")
print("Решение x:", x_computed1)
print("Ошибка относительно точного решения x*:", error1)

print("\nДля второй системы:")
print("Решение x:", x_computed2)