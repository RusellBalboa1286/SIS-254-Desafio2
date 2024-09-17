import numpy as np

# Definir el sistema de ecuaciones A*x = b
A = np.array([[0.52, 0.2, 0.25],
              [0.3, 0.5, 0.2],
              [0.18, 0.2, 0.55]])

b = np.array([4800, 5810, 5690])

# Definir los valores iniciales de x
x = np.zeros_like(b, dtype=np.double)

# Definir el número máximo de iteraciones
max_iterations = 100

# Tolerancia para la convergencia
tolerance = 1e-10

# Método de Jacobi
def jacobi(A, b, x, max_iterations, tolerance):
    D = np.diag(A)  # Diagonal de A
    R = A - np.diagflat(D)  # Resto de la matriz A
    
    for i in range(max_iterations):
        x_new = (b - np.dot(R, x)) / D
        
        # Verificar la convergencia (norma del error)
        error = np.linalg.norm(x_new - x, ord=np.inf)
        
        # Imprimir resultados de la iteración
        print(f"Iteración {i+1}: x = {x_new}, Error = {error}")
        
        if error < tolerance:
            print("Convergencia alcanzada.")
            break
        
        x = x_new
    
    return x

# Ejecutar el método de Jacobi
solution = jacobi(A, b, x, max_iterations, tolerance)

print("\nSolución final:")
print(solution)
