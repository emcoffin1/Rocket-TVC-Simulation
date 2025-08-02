import numpy as np
import SheetMethod as sh


def compute_IJ(XB, YB, XC, YC, S, phi):
    """
    Computes I and J (normal and tangential) integral components of velocity
    for use in source panel flow analysis
    INPUTS:
        - XB:   boundary points             :np.ndarray
        - YB:   boundary points             :np.ndarray
        - XC:   Control points              :np.ndarray
        - YC:   Control points              :np.ndarray
        - S:    Panel curve                 :np.ndarray
        - phi:  angle from positive x-axis
                to inside surface of panel  :np.ndarray

    OUTPUTS:
        - I
    """
    # Number of panels
    numPans = len(XB)

    # Initialize I and J
    I = np.zeros([numPans, numPans])
    J = np.zeros([numPans, numPans])

    # Iterate through each I for each J
    for i in range(numPans):
        for j in range(numPans):
            if j != i:
                A = -(XC[i] - XB[j]) * np.cos(phi[j]) - (YC[i] - YB[j]) * np.sin(phi[j])
                B = (XC[i] - XB[j])**2 + (YC[i] - YB[j])**2

                # C and D in normal (y)
                Cn = -np.sin(phi[j])
                Dn = (YC[i] - YB[j])

                # C and D in tangent (x)
                Ct = -np.cos(phi[j])
                Dt = (XC[i] - XB[j])

                E = np.sqrt(B - A**2)
                # E might be invalid -> set to zero
                if E == 0 or np.iscomplex(E) or np.isnan(E) or np.isinf(E):
                    I[i,j] = 0
                    J[i,j] = 0

                else:
                    # Compute I (for normal velocity)
                    term1 = 0.5 * Cn * np.log((S[j]**2 + 2*A*S[j] + B) / B)
                    term2 = ((Dn - A * Cn) / E) * (np.atan2((S[j] + A) / E) - np.atan2(A/E))
                    I[i,j] = term1 + term2

                    # Compute J (for tangential velocity)
                    term1 = 0.5 * Ct * np.log((S[j] ** 2 + 2 * A * S[j] + B) / B)
                    term2 = ((Dt - A * Ct) / E) * (np.atan2((S[j] + A) / E) - np.atan2(A / E))
                    J[i, j] = term1 + term2

            # Now that all values are computed, zero out any mistakes
            if np.isnan(I[i,j]) or np.isinf(I[i,j]) or np.iscomplex(I[i,j]):
                I[i,j] = 0
            if np.isnan(J[i,j]) or np.isinf(J[i,j]) or np.iscomplex(J[i,j]):
                J[i,j] = 0

    return I,J



