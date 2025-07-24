import pygame
import numpy as np
from scipy.spatial.transform import Rotation as R
from models.TVC.LqrQuaternionModels import QuaternionFinder, LQR

# --- Pygame Init ---
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 25)
running = True

# --- Simulation Setup ---
q = np.eye(3) * 0.5
r = np.eye(3) * 1
traj = np.array([0, 0, 0, 1])
att = np.array([0, 4, 0, 1])
loc = np.array([0, 0, 1])

thrust = 5000
p_y_inertia = 156.4
vehicle_height = 5.0
dt = 0.01
quat_tol = 1e-8
quat = QuaternionFinder(lqr=LQR(q=q, r=r))

rocket_pos = (WIDTH // 2, HEIGHT // 2)

while running:
    dt = clock.tick(60) / 1000  # fixed timestep
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Normalize quaternion and compute angular velocity
    att = att / np.linalg.norm(att)
    w = quat.getAngularVelocityCorrection(rocket_loc=loc, rocket_quat=att)

    # Calculate pitch correction (theta_y)
    theta_1 = p_y_inertia / (thrust * vehicle_height)
    theta_y = theta_1 * w[1] / dt
    theta_x = theta_1 * w[0] / dt  # yaw (around X)

    # Build rotation about Y axis only
    omega_vec = np.array([0.0, -theta_y, 0.0])
    rotation_increment = R.from_rotvec(omega_vec)

    current_rot = R.from_quat(att)
    new_rot = rotation_increment * current_rot
    att = new_rot.as_quat()

    # Extract pitch (rotation around Y-axis)
    euler = R.from_quat(att).as_euler('xyz', degrees=True)
    pitch = euler[1]  # Y-axis rotation
    yaw = euler[0]

    # --- Drawing ---
    screen.fill((20, 20, 30))
    pygame.draw.line(screen, (200, 200, 200), (WIDTH // 2, HEIGHT), (WIDTH // 2, 0), 2)

    # Rocket visualization
    rocket_len = 100
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)
    x0 = rocket_pos[0]
    y0 = rocket_pos[1]
    x1 = x0 + rocket_len * np.sin(pitch_rad)
    y1 = y0 - rocket_len * np.cos(pitch_rad)
    pygame.draw.line(screen, (255, 255, 255), (x0, y0), (x1, y1), 8)

    # Display pitch angle
    pitch_text = font.render(f"Pitch: {pitch:.2f}°", True, (255, 255, 255))
    yaw_text = font.render(f"Yaw:   {yaw:.2f}°", True, (255, 255, 255))
    screen.blit(pitch_text, (10, 10))
    screen.blit(yaw_text,   (10, 35))

    pygame.display.flip()

pygame.quit()




