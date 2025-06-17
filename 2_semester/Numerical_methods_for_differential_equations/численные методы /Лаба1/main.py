import numpy as np
import matplotlib.pyplot as plt

# Определение функции f(x, y)
def f(x, y):
    return (y / (x + 1)) - y**2  # Правая часть уравнения y' = (y / (x + 1)) - y^2

# Точное решение для сравнения
def exact_solution(x):
    return (2 * (x + 1)) / (x**2 + 2 * x + 2)

# Метод Рунге-Кутта 4-го порядка
def runge_kutta_4(f, x0, y0, h, n):
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0] = x0
    y[0] = y0
    
    for i in range(n):
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + h / 2, y[i] + k1 / 2)
        k3 = h * f(x[i] + h / 2, y[i] + k2 / 2)
        k4 = h * f(x[i] + h, y[i] + k3)
        
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x[i + 1] = x[i] + h
    
    return x, y

# Параметры задачи
x0, y0 = 0, 1  # Начальные условия
h = 0.5  # Шаг
n = int(1 / h)  # Количество шагов

# Решение с шагом h
x_h, y_h = runge_kutta_4(f, x0, y0, h, n)

# Решение с шагом h/2
h2 = h / 2
n2 = int(1 / h2)
x_h2, y_h2 = runge_kutta_4(f, x0, y0, h2, n2)

# Точное решение на общих точках
y_exact = exact_solution(x_h)

# Сравнение значений y(x) на общих точках
y_h2_common = y_h2[::2]  # Значения y_h2 на общих точках

# Вывод результатов
print("x \t y(h) \t y(h/2) \t y(exact) \t Разница (h) \t Разница (h/2)")
for i in range(len(x_h)):
    diff_h = abs(y_h[i] - y_exact[i])
    diff_h2 = abs(y_h2_common[i] - y_exact[i])
    print(f"{x_h[i]:.2f} \t {y_h[i]:.6f} \t {y_h2_common[i]:.6f} \t {y_exact[i]:.6f} \t {diff_h:.6f} \t {diff_h2:.6f}")

# График решений
plt.plot(x_h, y_h, 'o-', label=f"h = {h}")
plt.plot(x_h2, y_h2, 'x-', label=f"h/2 = {h2}")
plt.plot(x_h, y_exact, '--', label="Точное решение")
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('Решение задачи Коши методом Рунге-Кутта 4-го порядка')
plt.legend()
plt.grid()
plt.show()