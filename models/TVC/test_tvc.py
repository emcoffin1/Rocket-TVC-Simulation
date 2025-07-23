import numpy as np

from LqrQuaternionModels import *
from TVCModel import *

if __name__ == "__main__":

    q = np.eye(3) * 1
    r = np.eye(3) * 0.1
    lqr = LQR(q=q, r=r)
    quat = QuaternionFinder(lqr=lqr)

    angle_rotate_deg = 15
    angle_rotate = np.deg2rad(15)
    # Rotated on the y-axis
    # [0, 15deg, 0, 15deg]
    q_rocket = np.array([0.0, np.sin(angle_rotate/2), 0, np.cos(angle_rotate/2)])

    w = quat.getAngularVelocityCorrection(rocket=q_rocket, alt_m=0)
    print(w)





