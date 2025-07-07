from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

num = [1.0, 2.000, 5.000]
den = [7800, 0, 0, 0]

system = signal.TransferFunction(num, den)
w, mag, phase = signal.bode(system)

phase = (phase + 180) % 360 - 180  # Normalize

plt.subplot(2,1,1)
plt.semilogx(w, mag)
plt.subplot(2,1,2)
plt.semilogx(w, phase)
plt.show()
