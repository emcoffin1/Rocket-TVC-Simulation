import numpy as np
import matplotlib.pyplot as plt

# Vortex parameters
gamma = 4.0  # circulation strength (m^2/s)

# Grid in polar coordinates
r = np.linspace(0.05, 3.0, 300)
theta = np.linspace(0, 2 * np.pi, 300)
R, T = np.meshgrid(r, theta)

# Convert to Cartesian
X = R * np.cos(T)
Y = R * np.sin(T)

# Streamfunction for a point vortex
psi = -(gamma / (2 * np.pi)) * np.log(R)

# Velocity components (in polar)
Vr = np.zeros_like(R)  # no radial flow in a pure vortex
Vtheta = gamma / (2 * np.pi * R)

# Convert to Cartesian components
U = Vr * np.cos(T) - Vtheta * np.sin(T)
V = Vr * np.sin(T) + Vtheta * np.cos(T)

# Plotting
fig, ax = plt.subplots(figsize=(8, 8))
contours = ax.contour(X, Y, psi, levels=30, colors='blue', linewidths=0.7)
ax.quiver(X[::10, ::10], Y[::10, ::10], U[::10, ::10], V[::10, ::10], color='black')

# Mark the center
ax.plot(0, 0, 'ro', label='Vortex center')

ax.set_aspect('equal')
ax.set_title("Streamlines and Velocity Field of a Vortex")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


