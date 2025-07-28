import numpy as np

# neg angle
xl_p = np.array([1, 0, -1])
xf_p = np.array([0, 1, 0])
xl_n = np.array([-1, 0, -1])
xf_n = np.array([0, -1, 0])

xc_p = np.linalg.cross(xl_p, xf_p)
xc_n = np.linalg.cross(xl_n, xf_n)

print(xc_p)
print(xc_n)

xl_p = np.array([0, 1, -1])
xf_p = np.array([-1, 0, 0])
xl_n = np.array([0, -1, -1])
xf_n = np.array([1, 0, 0])

xc_p = np.linalg.cross(xl_p, xf_p)
xc_n = np.linalg.cross(xl_n, xf_n)

print(xc_p)
print(xc_n)
