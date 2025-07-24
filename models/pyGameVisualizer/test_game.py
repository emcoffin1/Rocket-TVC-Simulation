import os
# Skip real audio hardware probing
os.environ["SDL_AUDIODRIVER"] = "dummy"

import pygame
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from models.TVC.LqrQuaternionModels import QuaternionFinder, LQR

# — Init only display & font —
pygame.display.init()
pygame.font.init()

# — Pygame setup —
WIDTH, HEIGHT = 800, 600
screen  = pygame.display.set_mode((WIDTH, HEIGHT))
clock   = pygame.time.Clock()
font    = pygame.font.SysFont(None, 25)

# — LQR Quaternion controller setup —
q_mat = np.eye(3) * 0.1
r_mat = np.eye(3) * 1.0
ctl   = QuaternionFinder(lqr=LQR(q=q_mat, r=r_mat))

# — Constant vehicle parameters —
THRUST         = 5000.0    # N
MASS           = 1000.0    # kg
LEVER_DISTANCE =   3.0     # m from engine to CG
INERTIA        = np.array([156.4, 156.4, 156.4])  # Ixx, Iyy, Izz
GRAVITY        = np.array([0.0, 0.0, -9.81])      # m/s²

# — Fixed reference for LQR target —
ROCKET_LOC = np.array([0.0, 0.0, 1.0])

# — Initial state: start tilted “out” about Y-axis —
att = np.array([0.0, 4.0, 0.0, 1.0])  # quaternion [x, y, z, w]
pos = np.zeros(3)                     # [x, y, z] in meters
vel = np.zeros(3)                     # [vx, vy, vz] in m/s

# — Visual scale: meters → pixels —
SCALE = 50

running = True
while running:
    dt = clock.tick(60) / 1000.0  # seconds

    # — Quit event —
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False

    # — Attitude correction (LQR → angular‑velocity command) —
    att = att / np.linalg.norm(att)
    ω_cmd = ctl.getAngularVelocityCorrection(
        rocket_loc=ROCKET_LOC,
        rocket_quat=att
    )  # returns [ωx, ωy, ωz] in rad/s

    # Compute small gimbal angles (rad)
    θ_gain = INERTIA[1] / (THRUST * LEVER_DISTANCE)
    θx = θ_gain * ω_cmd[0] / dt
    θy = θ_gain * ω_cmd[1] / dt

    # Build & apply combined small rotation about X & Y
    rot_inc = R.from_rotvec(np.array([ θx, -θy, 0.0 ]))
    R_curr  = R.from_quat(att)
    R_new   = rot_inc * R_curr
    att     = R_new.as_quat()

    # — Translational physics under thrust & gravity —
    thrust_dir = R_new.apply([0.0, 1.0, 0.0])
    accel      = (THRUST * thrust_dir) / MASS + GRAVITY
    vel       += accel * dt
    pos       += vel   * dt

    # — Map 3D pos → 2D screen (X→right drift; Z altitude shown in text) —
    screen_x = (WIDTH  // 2) + pos[0] * SCALE
    screen_y = (HEIGHT // 2)               # rocket stays vertically centered

    # — Extract Euler for display —
    roll, pitch, yaw = R_new.as_euler('xyz', degrees=True)

    # — Drawing —
    screen.fill((20, 20, 30))
    # centerline
    pygame.draw.line(screen, (200,200,200),
                     (WIDTH//2, HEIGHT), (WIDTH//2, 0), 2)

    # Rocket “body” line pitched by pitch
    pr    = math.radians(pitch)
    x_end = screen_x + 100 * math.sin(pr)
    y_end = screen_y - 100 * math.cos(pr)
    pygame.draw.line(screen, (255,255,255),
                     (screen_x, screen_y), (x_end, y_end), 8)

    # — Overlay text —
    screen.blit(font.render(f"Pitch:    {pitch:.2f}°", True, (255,255,255)), (10, 10))
    screen.blit(font.render(f"Yaw:      {yaw:.2f}°", True, (255,255,255)), (10, 35))
    screen.blit(font.render(f"Altitude: {pos[2]:.2f} m", True, (255,255,255)), (10, 60))

    pygame.display.flip()

pygame.quit()
