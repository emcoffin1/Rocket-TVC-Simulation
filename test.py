import numpy as np

l_p = np.array([0.2302, 0, -2.06248])
f_p = np.array([0, -2.89, 0])
l_n = np.array([-0.2302, 0, -2.06248])
f_n = np.array([0, 2.89, 0])

c_p = np.linalg.cross(l_p, f_n)
c_n = np.linalg.cross(l_n, f_n)

print(c_p)
print(c_n)

exp = 0.2302 * 93.49
print(exp)

