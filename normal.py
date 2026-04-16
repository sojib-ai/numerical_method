import numpy as np

# 1. Define matrices A and b from your image
A = np.array([[2, 3],
    [4, 5],
    [6, 7]])

b = np.array([7, 11, 15])

# 2. Calculate A Transpose (A^T)
A_T = A.T

# 3. Compute (A^T * A)
# This results in a 2x2 square matrix
A_T_A = A_T @ A

# 4. Compute (A^T * b)
A_T_b = A_T @ b

# 5. Solve for x: x = (A^T * A)^-1 * (A^T * b)
# We use np.linalg.solve for better numerical precision than direct inversion
x = np.linalg.solve(A_T_A, A_T_b)

print("Solution for x1 and x2:")
print(f"x1 = {x[0]:.2f}")
print(f"x2 = {x[1]:.2f}")
