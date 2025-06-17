import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tabulate import tabulate

# Параметры задачи
a = 1.0 
l = 1.0  
T = 0.5 
M = 10   
N = 10  

# Шаги сетки
h = l / M
tau = T / N

# Коэффициент для разностной схемы
r = (a * tau / h) ** 2

# Инициализация сетки
u = np.zeros((M+1, N+1))

# Граничные условия
def gamma0(t):
    return -3 * t

def gamma1(t):
    return t ** 2

# Начальные условия
def phi(x):
    return 0.0

def psi(x):
    return 0.0

# Заполнение начальных условий
for m in range(M+1):
    x = m * h
    u[m, 0] = phi(x)  # u(x,0) = φ(x)

# Первый временной слой (n=1)
for m in range(1, M):
    x = m * h
  
    u[m, 1] = u[m, 0] + tau * psi(x) + 0.5 * r * (u[m-1, 0] - 2*u[m, 0] + u[m+1, 0])

# Граничные условия на первом временном слое
u[0, 1] = gamma0(tau)
u[M, 1] = gamma1(tau)

# Основной цикл по времени
for n in range(1, N):
    # Граничные условия
    u[0, n+1] = gamma0((n+1)*tau)
    u[M, n+1] = gamma1((n+1)*tau)
    
    # Внутренние узлы
    for m in range(1, M):
        u[m, n+1] = 2*u[m, n] - u[m, n-1] + r * (u[m-1, n] - 2*u[m, n] + u[m+1, n])


x = np.linspace(0, l, M+1)
t = np.linspace(0, T, N+1)
X, T = np.meshgrid(x, t)

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, u.T, cmap='viridis')

ax.set_xlabel('Пространство (x)')
ax.set_ylabel('Время (t)')
ax.set_zlabel('Смещение u(x,t)')
ax.set_title('Решение уравнения колебаний струны')

plt.tight_layout()
plt.show()

print("\nТаблица значений u(x,t):")
headers = ["x\\t"] + [f"t={tn:.2f}" for tn in t]
table_data = []
for m in range(M+1):
    row = [f"x={x[m]:.2f}"] + [f"{u[m,n]:.4f}" for n in range(N+1)]
    table_data.append(row)

print(tabulate(table_data, headers=headers, tablefmt="grid"))