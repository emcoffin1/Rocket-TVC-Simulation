import pygame
import math
from models.VehicleModels import Rocket, rk4_step, unpackStates
from models import EnvironmentalModels
from scipy.spatial.transform import Rotation as R
import numpy as np

# Setup
pygame.init()
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
firing = True
time = 0.0
print("starting")
while running:
    dt = clock.tick(60) / 1000

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    state = rk4_step(rocket, state, dt)
    pos, vel, quat, omega, mass, time, aoa, beta = unpackStates(state)
    force = rocket.thrust[-1]
    acc = force / mass
    # print(quat)


    # Rotation from quat
    rotation = R.from_quat(quat=quat)
    roll, pitch, yaw = rotation.as_euler("xyz", degrees=True)
    roll *= visual_scale
    pitch *= visual_scale
    yaw *= visual_scale


    # ===================== #
    # -- STATE VARIABLES -- #
    # ===================== #

    # Controls
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        thrust_angle += 60 * dt
    if keys[pygame.K_RIGHT]:
        thrust_angle -= 60 * dt

    firing = (True if rocket.engine.combustion_chamber.active else False)

    # Drawing
    screen.fill((20, 20, 30))

    # ========== #
    # -- TEXT -- #
    # ========== #

    # Altitude Text
    alt = round(pos[2]/1000, 3)
    alt_surface = font.render(f"ALT: {alt:.2f}km", True, (255, 255, 255))
    alt_text_rect = alt_surface.get_rect(center=(WIDTH//2, HEIGHT*0.95))
    screen.blit(alt_surface, alt_text_rect)

    # Time Text
    time_st = round(time, 2)
    t_surface = font.render(f"T + {time_st:.1f}", True, (255, 255, 255))
    t_text_rect = t_surface.get_rect(center=(WIDTH//2, HEIGHT*0.05))
    screen.blit(t_surface, t_text_rect)

    # Attitude
    roll_text = font.render(f"ROLL: {roll:.2f}", True, (255, 255, 255))
    pitch_text = font.render(f"PITCH: {pitch:.2f}", True, (255, 255, 255))
    yaw_text = font.render(f"YAW: {yaw:.2f}", True, (255, 255, 255))

    x = WIDTH * 0.01
    y = HEIGHT // 2

    screen.blit(roll_text, (x, y - 20))
    screen.blit(pitch_text, (x, y))  # Adjust spacing as needed
    screen.blit(yaw_text, (x, y + 20))

    # Location
    x_text = font.render(f"X: {pos[0]:.2f}", True, (255, 255, 255))
    y_text = font.render(f"Y: {pos[1]:.2f}", True, (255, 255, 255))

    x_text_rect = t_surface.get_rect(center=(WIDTH//2 - 100, HEIGHT*0.95))
    y_text_rect = t_surface.get_rect(center=(WIDTH//2 + 100, HEIGHT*0.95))

    screen.blit(x_text, x_text_rect)
    screen.blit(y_text, y_text_rect)

    # Acceleration/Velocity
    acc_text = font.render(f"ACC: {acc:.2f}m/s2", True, (255, 255, 255))
    vel_text = font.render(f"VEL: {np.linalg.norm(vel):.2f}m/s", True, (255, 255, 255))
    screen.blit(acc_text, (WIDTH*0.8, y+10))
    screen.blit(vel_text, (WIDTH*0.8, y-10))

    # ============= #
    # -- OBJECTS -- #
    # ============= #

    # Trajectory
    pygame.draw.line(screen, (200,200,200), (WIDTH//2, HEIGHT), (WIDTH//2, 0), 2)

    # Rocket
    r_len = -50
    pitch_rad = np.deg2rad(pitch)
    x0 = rocket_pos[0] + pos[0]
    y0 = rocket_pos[1]
    x1 = x0 + r_len * np.sin(pitch_rad)
    y1 = y0 + r_len * np.cos(pitch_rad)

    pygame.draw.line(screen, (255,255,255), (x0, y0), (x1, y1), 8)

    if firing == True:
        # Thrust vector
        t_len = 15
        angle_rad = math.radians(thrust_angle)
        end_x = rocket_pos[0] + t_len * math.sin(angle_rad)
        end_y = rocket_pos[1] + t_len * math.cos(angle_rad)
        pygame.draw.line(screen, (255,0,0), rocket_pos, (end_x, end_y), 5)


    time += dt
    pygame.display.flip()

pygame.quit()
