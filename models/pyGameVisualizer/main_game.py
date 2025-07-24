import os
# Skip real audio hardware probing
os.environ["SDL_AUDIODRIVER"] = "dummy"

import pygame
import math
from models.VehicleModels import Rocket, rk4_step, unpackStates
from models import EnvironmentalModels
from scipy.spatial.transform import Rotation as R
import numpy as np

# --- Only init what we need ---
pygame.display.init()
pygame.font.init()

# Setup
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 25)

print("init rockets")
rocket = Rocket()
state = rocket.state.copy()

# Init Rocket Location
rocket_pos = (WIDTH // 2, HEIGHT // 2)
thrust_angle = 0
visual_scale = 1

running = True
time = 0.0
print("starting")
while running:
    dt = clock.tick(60) / 1000

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Physics update
    state = rk4_step(rocket, state, dt)
    pos, vel, quat, omega, mass, time, aoa, beta = unpackStates(state)
    force = rocket.thrust[-1]
    acc = force / mass

    # Attitude extraction
    rotation = R.from_quat(quat)
    roll, pitch, yaw = rotation.as_euler("xyz", degrees=True)
    roll  *= visual_scale
    pitch *= visual_scale
    yaw   *= visual_scale

    # Controls
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        thrust_angle += 60 * dt
    if keys[pygame.K_RIGHT]:
        thrust_angle -= 60 * dt

    firing = rocket.engine.combustion_chamber.active

    # Drawing
    screen.fill((20, 20, 30))

    # — Altitude text —
    alt = pos[2] / 1000.0
    alt_surf = font.render(f"ALT: {alt:.2f} km", True, (255, 255, 255))
    alt_rect = alt_surf.get_rect(center=(WIDTH//2, HEIGHT * 0.95))
    screen.blit(alt_surf, alt_rect)

    # — Time text —
    t_surf = font.render(f"T + {time:.1f}", True, (255, 255, 255))
    t_rect = t_surf.get_rect(center=(WIDTH//2, HEIGHT * 0.05))
    screen.blit(t_surf, t_rect)

    # — Attitude text —
    x_off = WIDTH * 0.01
    y_mid = HEIGHT // 2
    roll_surf  = font.render(f"ROLL:  {roll:.2f}",  True, (255,255,255))
    pitch_surf = font.render(f"PITCH: {pitch:.2f}", True, (255,255,255))
    yaw_surf   = font.render(f"YAW:   {yaw:.2f}",   True, (255,255,255))
    screen.blit(roll_surf,  (x_off, y_mid - 20))
    screen.blit(pitch_surf, (x_off, y_mid))
    screen.blit(yaw_surf,   (x_off, y_mid + 20))

    # — X/Y position text —
    x_surf = font.render(f"X: {pos[0]:.2f}", True, (255,255,255))
    y_surf = font.render(f"Y: {pos[1]:.2f}", True, (255,255,255))
    x_rect = x_surf.get_rect(center=(WIDTH//2 - 100, HEIGHT * 0.95))
    y_rect = y_surf.get_rect(center=(WIDTH//2 + 100, HEIGHT * 0.95))
    screen.blit(x_surf, x_rect)
    screen.blit(y_surf, y_rect)

    # — Acceleration / Velocity text —
    acc_surf = font.render(f"ACC: {acc:.2f} m/s²", True, (255,255,255))
    vel_surf = font.render(f"VEL: {np.linalg.norm(vel):.2f} m/s", True, (255,255,255))
    screen.blit(acc_surf, (WIDTH * 0.8, y_mid + 10))
    screen.blit(vel_surf, (WIDTH * 0.8, y_mid - 10))

    # — Trajectory centerline —
    pygame.draw.line(screen, (200,200,200), (WIDTH//2, HEIGHT), (WIDTH//2, 0), 2)

    # — Rocket body line —
    r_len     = -50
    pitch_rad = math.radians(pitch)
    x0        = rocket_pos[0] + pos[0]
    y0        = rocket_pos[1]
    x1        = x0 + r_len * math.sin(pitch_rad)
    y1        = y0 + r_len * math.cos(pitch_rad)
    pygame.draw.line(screen, (255,255,255), (x0, y0), (x1, y1), 8)

    # — Thrust vector if firing —
    if firing:
        t_len = 15
        a_rad = math.radians(thrust_angle)
        end_x = rocket_pos[0] + t_len * math.sin(a_rad)
        end_y = rocket_pos[1] + t_len * math.cos(a_rad)
        pygame.draw.line(screen, (255,0,0), rocket_pos, (end_x, end_y), 5)

    # Update display & time
    pygame.display.flip()
    time += dt

pygame.quit()
