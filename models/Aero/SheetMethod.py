import numpy as np


def circleGeneration(aoad: float = 0.0, numb: int = 9):
    """
    Creates a circle of panels as well as the panel geometry,
    change later to either create circle or handle other geometry
    --math stays the same regardless of shape--
    INPUTS:
        - aoad: angle of attack of freestream velocity :float = 0.0
        - numb: desired number of panels               :int = 9
    OUTPUTS:
        - XB:   boundary points             :np.ndarray
        - YB:   boundary points             :np.ndarray
        - XC:   Control points              :np.ndarray
        - YC:   Control points              :np.ndarray
        - S:    Panel curve                 :np.ndarray
        - phi:  angle from positive x-axis
                to inside surface of panel  :np.ndarray
        - beta: angle between freestream
                velocity and panel normal
                vector                      :np.ndarray

    """
    AoAd = aoad
    AoAr = np.deg2rad(AoAd)
    numB = numb
    t0 = (360/(numB-1))/2

    theta = np.linspace(0,360,numB)
    theta = theta + t0
    theta = np.deg2rad(theta)

    XB = np.cos(theta)
    YB = np.sin(theta)

    numPan = len(XB) - 1

    edge = np.zeros(numPan)
    for i in range(numPan):
        edge[i] = (XB[i+1] - XB[i]) * (YB[i+1] - YB[i])

    # Determine direction
    sumEdge = np.sum(edge)
    if sumEdge < 0:
        XB = np.flipud(XB)
        YB = np.flipud(YB)


    XC = np.zeros(numPan)
    YC = np.zeros(numPan)
    S = np.zeros(numPan)
    phi = np.zeros(numPan)

    for i in range(numPan):
        XC[i] = 0.5 * (XB[i] + XB[i+1])
        YC[i] = 0.5 * (YB[i] + YB[i+1])
        dx = XB[i+1] - XB[i]
        dy = YB[i+1] - YB[i]
        S[i] = np.sqrt(dx**2 + dy**2)
        phi[i] = np.atan2(dy, dx)
        if phi[i] < 0:
            phi[i] = phi[i] + 2 * np.pi

    delta = phi + (np.pi / 2)
    beta  = delta - AoAr
    beta[beta > 2*np.pi] = beta[beta > 2 * np.pi] - 2*np.pi

    return XB, YB, XC, YC, S, phi, beta


