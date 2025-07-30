import numpy as np


F = np.array([100, 0 , -200])

r = np.array([0, 0, 1])

tor = np.linalg.cross(r,F)
print(tor)
