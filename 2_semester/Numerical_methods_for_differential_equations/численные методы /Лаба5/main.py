import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

print("Решение уравнения методом коллокации:")
print("u'' + (2u')/(x - 2) + u(x - 2) = 1")
print("Граничные условия: u(0) = -0.5, u(1) = -1")
print("Точное решение: u(x) = 1 / (x - 2)")

# Символьные переменные
x = sp.Symbol('x')
c0, c1, c2, c3 = sp.symbols('c0 c1 c2 c3')

# Приближённое решение (полином 3 степени)
u = c0 + c1 * x + c2 * x ** 2 + c3 * x ** 3

# Производные
u_prime = sp.diff(u, x)
u_double_prime = sp.diff(u_prime, x)

# Правая часть уравнения
f = 1

# Остаток (невязка)
R = u_double_prime + (2 * u_prime / (x - 2)) + u * (x - 2) - f

# Граничные условия
eq1 = u.subs(x, 0) + 0.5     # u(0) = -0.5
eq2 = u.subs(x, 1) + 1       # u(1) = -1

# Точки коллокации внутри интервала (не граничные)
collocation_points = [0.3, 0.7]
eq3 = R.subs(x, collocation_points[0])
eq4 = R.subs(x, collocation_points[1])

# Список уравнений и переменных
equations = [eq1, eq2, eq3, eq4]
variables = [c0, c1, c2, c3]

# Построение СЛАУ: A * coeffs = b
A = np.zeros((4, 4))
b = np.zeros(4)

for i, eq in enumerate(equations):
    for j, var in enumerate(variables):
        A[i, j] = float(eq.coeff(var)) if var in eq.free_symbols else 0
    b[i] = -float(eq.subs({v: 0 for v in variables}))

# Решаем СЛАУ
coeffs = np.linalg.solve(A, b)
c0_val, c1_val, c2_val, c3_val = coeffs

# Вывод коэффициентов
print("\nКоэффициенты приближённого решения:")
print(f"c0 = {c0_val:.6f}")
print(f"c1 = {c1_val:.6f}")
print(f"c2 = {c2_val:.6f}")
print(f"c3 = {c3_val:.6f}")

# Приближённая функция
def u_approx(x_val):
    return c0_val + c1_val * x_val + c2_val * x_val ** 2 + c3_val * x_val ** 3

# Точное решение
def u_exact(x_val):
    return 1 / (x_val - 2)

# Сравнение на отдельных точках
print("\nСравнение приближённого и точного решений:")
points = [0, 0.25, 0.5, 0.75, 1]
for p in points:
    approx = u_approx(p)
    exact = u_exact(p)
    error = approx - exact
    print(f"x = {p:.2f}: приближ. = {approx:.6f}, точн. = {exact:.6f}, ошибка = {error:.6f}")

# Графики
x_vals = np.linspace(0, 1, 200)
u_approx_vals = u_approx(x_vals)
u_exact_vals = u_exact(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, u_approx_vals, label='Приближённое решение', color='blue')
plt.plot(x_vals, u_exact_vals, label='Точное решение', color='red', linestyle='--')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Сравнение приближённого и точного решений')
plt.legend()
plt.grid(True)
plt.show()

# Максимальная ошибка
error = np.abs(u_exact_vals - u_approx_vals)
print(f"\nМаксимальная ошибка: {np.max(error):.6f}")
