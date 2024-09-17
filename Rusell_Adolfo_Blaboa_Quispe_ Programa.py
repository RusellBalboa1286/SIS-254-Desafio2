import numpy as np

# Términos independientes
b = np.array([7.85, -19.3, 71.4])

# Valores iniciales
x = np.array([0.0, 0.0, 0.0])

# Matriz A
A = np.array([
    [3, -0.1, -0.2],
    [0.1, 7, -0.3],
    [0.3, 0.2, 10]
])

# Tolerancia y número máximo de iteraciones
tolerance = 1e-5
max_iter = 100

# Iteración Gauss-Seidel
for k in range(max_iter):
    x_old = np.copy(x)
    
    x[0] = (b[0] - A[0, 1]*x[1] - A[0, 2]*x[2]) / A[0, 0]
    x[1] = (b[1] - A[1, 0]*x[0] - A[1, 2]*x[2]) / A[1, 1]
    x[2] = (b[2] - A[2, 0]*x[0] - A[2, 1]*x[1]) / A[2, 2]
    
    # Verificar la convergencia
    if np.linalg.norm(x - x_old, np.inf) < tolerance:
        print(f"Convergió en {k+1} iteraciones")
        break

# Mostrar la solución
print("La solución aproximada es:")
print(f"x1 = {x[0]:.4f}")
print(f"x2 = {x[1]:.4f}")
print(f"x3 = {x[2]:.4f}")
