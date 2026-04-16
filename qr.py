import numpy as np

# 1. Define Matrix A and Vector b
A = np.array([[1, 0, 1],
    [1, 1, 0],
    [0, 1, 1]])

b = np.array([2, 1, 3])

# 2. Perform QR Decomposition
Q, R = np.linalg.qr(A)

# 3. Solve the system Rx = Q^T b
d = np.dot(Q.T, b)
x = np.linalg.solve(R, d)

# 4. Display results with full precision (no rounding)
print("Solution (exact floating point):")
print(f"x1 = {x[0]:.10f}")
print(f"x2 = {x[1]:.10f}")
print(f"x3 = {x[2]:.10f}")

# 5. Optional: Check if solution is very close to integers
x_rounded = np.round(x)
if np.allclose(x, x_rounded, atol=1e-10):
    print("\nThe solution appears to be integers:")
    print(f"x1 = {int(x_rounded[0])}")
    print(f"x2 = {int(x_rounded[1])}")
    print(f"x3 = {int(x_rounded[2])}")

else:
    print("\nThe solution is not integer. Use the floating point values above.")
