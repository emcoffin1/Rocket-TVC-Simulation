import numpy as np

from LqrQuaternionModels import *

if __name__ == "__main__":

    # q = np.eye(3) * 1
    # r = np.eye(3) * 0.1
    # lqr = LQR(q=q, r=r)
    # quat = QuaternionFinder(lqr=lqr)
    #
    # angle_rotate_deg = 15
    # angle_rotate = np.deg2rad(15)
    # # Rotated on the y-axis
    # # [0, 15deg, 0, 15deg]
    # q_rocket = np.array([0.0, np.sin(angle_rotate/2), 0, np.cos(angle_rotate/2)])
    #
    # w = quat.getAngularVelocityCorrection(rocket=q_rocket, alt_m=0)
    # print(w)
    #

    p_r = np.array([1,3,0])
    p_t = np.array([0,0,0])

    p_e = p_t - p_r
    p_e = p_e / np.linalg.norm(p_e)

    v_body = np.array([0,0,-1])
    v = np.cross(v_body, p_e)
    c = np.dot(v_body, p_e)

    s = np.sqrt((1+c)*2)
    q = np.array([v[0]/s, v[1]/s, v[2]/s, s/2])
    print(q)



