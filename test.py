import numpy as np
from scipy.linalg import solve_continuous_are

Q = np.diag([10, 10, 10, 5, 5])
R = np.eye(3) * 0.5

A = np.zeros((5, 5))
B = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 1.0],
], dtype=np.float64)

ctrb = np.hstack([B, A @ B, A @ A @ B, A @ A @ A @ B, A @ A @ A @ A @ B])
print("Rank of controllability matrix:", np.linalg.matrix_rank(ctrb, tol=1e-12))

P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P

print("Gain K:")
print(K)
