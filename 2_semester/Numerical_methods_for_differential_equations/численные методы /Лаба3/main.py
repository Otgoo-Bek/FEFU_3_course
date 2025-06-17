import numpy as np
import matplotlib.pyplot as plt

# Параметры задачи
a = 1.0  # Коэффициент теплопроводности
l = 1.0  # Длина области по x
T = 0.1  # Время моделирования
M = 10   # Количество шагов по пространству
N = 100  # Количество шагов по времени
h = l / M  # Шаг по пространству
tau = T / N  # Шаг по времени

# Сетка
x = np.linspace(0, l, M+1)  # Пространственная сетка
t = np.linspace(0, T, N+1)  # Временная сетка

# Начальное условие
def psi(x):
    return 3 * (x**2 - x)

# Источник тепла
def phi(x, t):
    return 2 * x * (1 - x)

# Граничные условия
def gamma0(t):
    return 0.0

def gamma1(t):
    return 0.0

# Инициализация сеток для всех методов
u_explicit = np.zeros((M+1, N+1))  # Явный метод
u_implicit = np.zeros((M+1, N+1))  # Чисто неявная схема
u_crank = np.zeros((M+1, N+1))     # Схема Кранка-Николсона

# Начальные условия
u_explicit[:, 0] = psi(x)
u_implicit[:, 0] = psi(x)
u_crank[:, 0] = psi(x)

# Явный метод
for n in range(N):
    for m in range(1, M):
        u_explicit[m, n+1] = u_explicit[m, n] + (a * tau / h**2) * (u_explicit[m+1, n] - 2 * u_explicit[m, n] + u_explicit[m-1, n]) + tau * phi(x[m], t[n])
    # Граничные условия
    u_explicit[0, n+1] = gamma0(t[n+1])
    u_explicit[M, n+1] = gamma1(t[n+1])

    # Вывод значений на каждом шаге по времени
    if (n+1) % (N // 6) == 0:  # Выводим каждые 6 шагов
        print(f"Явный метод, t = {t[n+1]:.4f}")
        print(u_explicit[:, n+1])

# Чисто неявная схема
for n in range(N):
    # Создание матрицы A и вектора b
    A = np.zeros((M+1, M+1))
    b = np.zeros(M+1)
    
    # Заполнение матрицы A и вектора b
    for m in range(1, M):
        A[m, m-1] = -a * tau / h**2
        A[m, m] = 1 + 2 * a * tau / h**2
        A[m, m+1] = -a * tau / h**2
        b[m] = u_implicit[m, n] + tau * phi(x[m], t[n+1])
    
    # Граничные условия
    A[0, 0] = 1
    b[0] = gamma0(t[n+1])
    A[M, M] = 1
    b[M] = gamma1(t[n+1])
    
    # Решение системы уравнений
    u_implicit[:, n+1] = np.linalg.solve(A, b)

    # Вывод значений на каждом шаге по времени
    if (n+1) % (N // 6) == 0:  # Выводим каждые 6 шагов
        print(f"Чисто неявная схема, t = {t[n+1]:.4f}")
        print(u_implicit[:, n+1])

# Схема Кранка-Николсона
for n in range(N):
    # Создание матрицы A и вектора b
    A = np.zeros((M+1, M+1))
    b = np.zeros(M+1)
    
    # Заполнение матрицы A и вектора b
    for m in range(1, M):
        A[m, m-1] = -a * tau / (2 * h**2)
        A[m, m] = 1 + a * tau / h**2
        A[m, m+1] = -a * tau / (2 * h**2)
        b[m] = (a * tau / (2 * h**2)) * u_crank[m-1, n] + (1 - a * tau / h**2) * u_crank[m, n] + (a * tau / (2 * h**2)) * u_crank[m+1, n] + (tau / 2) * (phi(x[m], t[n]) + phi(x[m], t[n+1]))
    
    # Граничные условия
    A[0, 0] = 1
    b[0] = gamma0(t[n+1])
    A[M, M] = 1
    b[M] = gamma1(t[n+1])
    
    # Решение системы уравнений
    u_crank[:, n+1] = np.linalg.solve(A, b)

    # Вывод значений на каждом шаге по времени
    if (n+1) % (N // 6) == 0:  # Выводим каждые 6 шагов
        print(f"Схема Кранка-Николсона, t = {t[n+1]:.4f}")
        print(u_crank[:, n+1])

# Визуализация результатов
plt.figure(figsize=(18, 6))

# Явный метод
plt.subplot(1, 3, 1)
for n in range(0, N+1, N//6):
    plt.plot(x, u_explicit[:, n], label=f't={t[n]:.4f}')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.title('Явный метод')
plt.legend()
plt.grid()

# Чисто неявная схема
plt.subplot(1, 3, 2)
for n in range(0, N+1, N//6):
    plt.plot(x, u_implicit[:, n], label=f't={t[n]:.4f}')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.title('Чисто неявная схема')
plt.legend()
plt.grid()

# Схема Кранка-Николсона
plt.subplot(1, 3, 3)
for n in range(0, N+1, N//6):
    plt.plot(x, u_crank[:, n], label=f't={t[n]:.4f}')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.title('Схема Кранка-Николсона')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()