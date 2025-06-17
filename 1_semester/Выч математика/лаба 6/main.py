import numpy as np

def simple_iteration_method(matrix, initial_vector, tolerance):
    n = matrix.shape[0]
    vector = initial_vector / np.linalg.norm(initial_vector)  
    eigenvalue = 0
    iterations = 0

    while True:
        y = matrix @ vector  
        vector_next = y / np.linalg.norm(y)  

        
        eigenvalue_next = np.dot(vector_next, matrix @ vector_next) / np.dot(vector_next, vector_next)
        iterations += 1


        if np.abs(eigenvalue_next - eigenvalue) <= tolerance:
            break
        
        vector = vector_next  
        eigenvalue = eigenvalue_next  

    return eigenvalue, iterations


def rotation_method(matrix, tolerance):
    n = matrix.shape[0]
    modified_matrix = matrix.copy()  
    iterations = 0

    while True:
        max_off_diagonal = 0  
        row, col = -1, -1  
        
        
        for r in range(n):
            for c in range(r + 1, n):
                if abs(modified_matrix[r, c]) > max_off_diagonal:
                    max_off_diagonal = abs(modified_matrix[r, c])
                    row, col = r, c

        if max_off_diagonal < tolerance:
            break

        elem_ii = modified_matrix[row, row]
        elem_jj = modified_matrix[col, col]
        elem_ij = modified_matrix[row, col]

        angle = 0.5 * np.arctan2(2 * elem_ij, elem_ii - elem_jj) if elem_ii != elem_jj else np.pi / 4

        rotation_matrix = np.eye(n)
        rotation_matrix[row, row] = np.cos(angle)
        rotation_matrix[col, col] = np.cos(angle)
        rotation_matrix[row, col] = -np.sin(angle)
        rotation_matrix[col, row] = np.sin(angle)

        modified_matrix = rotation_matrix.T @ modified_matrix @ rotation_matrix
        iterations += 1

    return np.diag(modified_matrix), iterations  


def generate_positive_definite_matrix(size):
    random_matrix = np.random.rand(size, size) 
    return (random_matrix + random_matrix.T) / 2 + size * np.eye(size) 


matrix_sizes = [3, 5, 7]  
precision_levels = [1e-3, 1e-7]  

for size in matrix_sizes:
    random_matrix = generate_positive_definite_matrix(size) 
    print(f"Матрица A ({size}x{size}):\n{random_matrix}\n")  
    
    for tolerance in precision_levels:
        initial_vector = np.random.rand(size)
        eigenvalue_iter, iterations_iter = simple_iteration_method(random_matrix, initial_vector, tolerance)
        eigenvalues_rot, iterations_rot = rotation_method(random_matrix.copy(), tolerance)
            
        print(f"ε: {tolerance}")
        print(f"Метод простых итераций:\nСобственное значение: {eigenvalue_iter},\nИтерации: {iterations_iter}")
        print(f"Метод вращений:\nСобственные значения: {eigenvalues_rot},\nИтерации: {iterations_rot}\n")