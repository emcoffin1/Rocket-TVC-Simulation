import numpy as np

def solve_for_circulation(I, Vinf, beta):
    """
    Solves for the circulation strength of each panel using matrix algebra
    Sum the final values to ensure close to zero
    INPUTS:
        - I:    integral component of normal velocity   :np.ndarray
        - Vinf: freestream velocity                     :float
        - beta: angle between Vinf (AoA)
                and panel normal vector                 :np.ndarray
    OUTPUTS:
        - Circulation strength of each panel            :np.ndarray
    """
    numPan = len(I)
    A = np.zeros([numPan, numPan])
    for i in range(numPan):
        for j in range(numPan):
            # Diagonal part is pi
            if (i == j):
                A[i,j] = np.pi

            else:
                A[i,j] = I[i,j]

    b = np.zeros(numPan)
    for i in range(numPan):
        b[i] = -Vinf * 2 * np.pi * np.cos(beta[i])

    strength = np.linalg.solve(A,b)

    return strength


def compute_velocities(J, Vinf, beta, strength):
    """"""
    numPan = len(beta)
    # Tangential velocity array
    Vt = np.zeros(numPan)
    # Pressure coefficient array
    Cp = np.zeros(numPan)

    for i in range(numPan):
        addVal = 0
        for j in range(numPan):
            addVal = addVal + (strength[j] / (2*np.pi)) * J[i,j]

        Vt[i] = Vinf*np.sin(beta[i]) + addVal
        Cp[i] = 1 - (Vt[i] / Vinf)**2




