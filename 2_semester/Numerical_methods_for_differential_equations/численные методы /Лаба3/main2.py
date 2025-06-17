import numpy as np
import matplotlib.pyplot as plt

def solve_bvp(N, p, q, f, alpha0, beta0, gamma0, alpha1, beta1, gamma1):
    if N <= 1:
        raise ValueError("Число узлов N должно быть больше 1.")

    h = 1 / N  # Шаг сетки
    x = np.linspace(0, 1, N + 1)  # Узлы сетки

    # Инициализация коэффициентов
    a = np.zeros(N + 1)
    b = np.zeros(N + 1)
    c = np.zeros(N + 1)
    d = np.zeros(N + 1)

    # Заполнение коэффициентов для внутренних узлов
    for i in range(1, N):
        r = p(x[i]) * h / 2
        modr = np.abs(r)
        a[i] = (1 / h**2) * (np.exp(modr) / (modr + 1) - r)  # Коэффициент a[i]
        c[i] = (1 / h**2) * (np.exp(modr) / (modr + 1) + r)  # Коэффициент c[i]
        b[i] = -a[i] - c[i] + q(x[i])  # Диагональный элемент
        d[i] = f(x[i])  # Правая часть

    # Граничные условия на левом конце (x = 0)
    if beta0 == 0:
        b[0] = alpha0
        d[0] = gamma0
        a[0] = 0
        c[0] = 0
    else:
        b[0] = alpha0 - beta0 / h
        c[0] = beta0 / h
        d[0] = gamma0
        a[0] = 0

    # Граничные условия на правом конце (x = 1)
    if beta1 == 0:
        b[N] = alpha1
        d[N] = gamma1
        a[N] = 0
        c[N] = 0
    else:
        b[N] = alpha1 + beta1 / h
        a[N] = -beta1 / h
        d[N] = gamma1
        c[N] = 0

    # Прямой ход метода прогонки
    for i in range(1, N + 1):
        factor = a[i] / b[i - 1]
        b[i] -= factor * c[i - 1]
        d[i] -= factor * d[i - 1]

    # Обратный ход метода прогонки
    y = np.zeros(N + 1)
    y[N] = d[N] / b[N]

    for i in range(N - 1, -1, -1):
        y[i] = (d[i] - c[i] * y[i + 1]) / b[i]

    return x, y, a, b, c, d

# Заданные функции и параметры
p = lambda x: -2  # Функция p(x)
q = lambda x: -1  # Функция q(x)
f = lambda x: (2 / (x + 1)**3) * np.exp(x)  # Функция f(x)

# Параметры задачи
N = 100  # Увеличиваем число узлов для повышения точности
alpha0, beta0, gamma0 = 1, 0, 1  # Граничные условия на левом конце
alpha1, beta1, gamma1 = 0, 1, np.exp(1) / 4  # Граничные условия на правом конце

# Решение задачи
x, y, a, b, c, d = solve_bvp(N, p, q, f, alpha0, beta0, gamma0, alpha1, beta1, gamma1)

# Точное решение
exact_solution = lambda x: np.exp(x) / (x + 1)

# Вывод результатов
print("\nРешение краевой задачи методом конечных разностей")
print("-------------------------------------------------")
print(f"Число узлов: {N}")
print(f"Шаг сетки: {1 / N:.4f}")
print("\nЗначения решения в узлах:")
print("{:<10} {:<15} {:<15}".format("x", "u(x)", "Ошибка"))
print("-" * 40)
error = np.abs(y - exact_solution(x))
for xi, yi, ei in zip(x, y, error):
    print(f"{xi:<10.4f} {yi:<15.6f} {ei:<15.6f}")

# Визуализация результатов
plt.figure(figsize=(8, 5))
plt.plot(x, y, 'b-', label='Численное решение')
plt.plot(x, exact_solution(x), 'r--', label='Точное решение')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Сравнение численного и точного решения')
plt.legend()
plt.grid()
plt.show()