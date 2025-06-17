import numpy as np

def compute_matrix_properties(matrices):
    properties = []
    for index, matrix in enumerate(matrices):
        norm = np.linalg.norm(matrix, ord=1) / matrix.shape[0] 
        condition_number = norm * np.linalg.norm(np.linalg.inv(matrix), ord=np.inf)
        properties.append((index + 1, norm, condition_number))
    return properties

def display_matrix(matrix, title):
    print(f"{title}:")
    print(np.array2string(matrix, precision=3, suppress_small=True, max_line_width=100))
    print("\n" + "=" * 50 + "\n")

def main():
    SIZE = 25
    matrices = [np.random.rand(SIZE, SIZE) for _ in range(5)]

    # Вычисление свойств матриц
    matrix_properties = compute_matrix_properties(matrices)
    
    for index, norm, cond in matrix_properties:
        display_matrix(matrices[index - 1], f"Матрица {index}")
        print(f"Норма матрицы {index}: {norm:.4f}")
        print(f"Число обусловленности матрицы {index}: {cond:.4f}")
        print("\n" + "=" * 50 + "\n")

    # Создание матрицы Вандермонда
    x = np.linspace(0, 1, SIZE)
    vandermonde_matrix = np.vander(x, increasing=True)
    b = np.ones(SIZE)

    # Решение системы уравнений
    vandermonde_solution = np.linalg.solve(vandermonde_matrix, b)
    cond_vandermonde = np.linalg.cond(vandermonde_matrix)

    display_matrix(vandermonde_matrix, "Матрица Вандермонда")
    print(f"Число обусловленности матрицы Вандермонда: {cond_vandermonde:.4f}")

    # Норма вектора b
    vector_norm = np.linalg.norm(b)
    print(f"Норма вектора b: {vector_norm:.4f}")
if __name__ == "__main__":
    main()