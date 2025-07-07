import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy import signal
import matplotlib.pyplot as plt
import control as ctrl

class Controller:
    def __init__(self, kp=20.0, kd=8.0):
        # Proportional Gain
        self.kp = kp
        # Derivative (damping) Gain
        self.kd = kd

    def control_torque(self, quat, omega, target_quat):
        # Convert to rotation error: q_err = q_desired^-1 * q_actual
        q_current = R.from_quat(quat)
        q_target = R.from_quat(target_quat)
        q_err = q_target.inv() * q_current

        # Convert error quaternion to rotation vector
        axis_angle = q_err.as_rotvec()
        torque_p = -self.kp * axis_angle

        # Derivative (damping) term
        torque_d = -self.kd * omega

        # Total control torque
        return torque_p + torque_d


class BodePlot:
    def __init__(self, kp, ki, kd, inertia):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.inertia = inertia

    def compute(self):
        self.num = [self.kd, self.kp, self.ki]
        self.den = [self.inertia, 0, 0, 0]
        print(f"TF Numerator: {self.num}")
        print(f"TF Denominator: {self.den}")
        return signal.TransferFunction(self.num, self.den)

    def graph(self):
        system = self.compute()
        w, mag, phase = signal.bode(system)

        # Normalize phase
        phase = (phase + 180) % 360 - 180

        # Print margins
        self.comments(w, mag, phase)

        plt.figure()
        plt.subplot(2,1,1)
        plt.semilogx(w, mag)
        plt.ylabel("Magnitude (dB)")
        plt.grid()

        plt.subplot(2,1,2)
        plt.semilogx(w, phase)
        plt.ylabel("Phase (deg)")
        plt.xlabel("Frequency (rad/s)")
        plt.grid()

        plt.tight_layout()
        plt.show()

    def comments(self, w, mag, phase):
        # Gain and phase margin calculation
        idx_mag_cross = np.where(np.diff(np.sign(mag)))[0]
        if len(idx_mag_cross) > 0:
            i = idx_mag_cross[0]
            wp = w[i]
            pm = 180 + phase[i]
            print(f"Phase Margin: {pm:.2f} deg at {wp:.2f} rad/s")
        else:
            print("Phase Margin: Not found")

        idx_phase_cross = np.where(np.diff(np.sign(phase + 180)))[0]
        if len(idx_phase_cross) > 0:
            j = idx_phase_cross[0]
            wg = w[j]
            gm = -mag[j]
            print(f"Gain Margin: {gm:.2f} dB at {wg:.2f} rad/s")
        else:
            print("Gain Margin: Not found")

    def run(self):
        self.graph()


import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

class ControlBodePlot:
    def __init__(self, kp, ki, kd, tau_d, inertia):
        # PID gains and derivative filter time constant
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.tau_d = tau_d  # Derivative filter time constant
        self.inertia = inertia  # Moment of inertia for the plant

        # Create controller and plant transfer functions
        self.controller = self.build_pid_with_filter()
        self.plant = ctrl.TransferFunction([1], [inertia, 0, 0])
        self.open_loop = self.controller * self.plant

        # Generate the plots
        self.plot_bode_and_step()

    def build_pid_with_filter(self):
        """
        Builds a PID controller with a filtered derivative term:
        C(s) = Kp + Ki/s + (Kd * s) / (τd * s + 1)
        """
        # Proportional term
        P = ctrl.TransferFunction([self.kp], [1])

        # Integral term
        I = ctrl.TransferFunction([self.ki], [1, 0])

        # Derivative term with first-order low-pass filter
        D = ctrl.TransferFunction([self.kd, 0], [self.tau_d, 1])

        # Total PID controller with filtered derivative
        return P + I + D

    def compute_step_response(self):
        """
        Computes the step response of the closed-loop system.
        """
        closed_loop = ctrl.feedback(self.open_loop, 1)
        t, y = ctrl.step_response(closed_loop)
        return t, y

    def plot_bode_and_step(self):
        # Define frequency range for analysis
        omega = np.logspace(-2, 2, 1000)

        # Compute magnitude and phase
        mag, phase, _ = ctrl.frequency_response(self.open_loop, omega)
        mag_db = 20 * np.log10(np.abs(mag).squeeze())
        phase_deg = np.angle(phase, deg=True).squeeze()

        # Calculate gain and phase margins
        gm, pm, wg, wp = ctrl.margin(self.open_loop)

        # Print margin values
        if gm and wg:
            print(f"Gain Margin: {20 * np.log10(gm):.2f} dB at {wg:.2f} rad/s")
        else:
            print("Gain Margin: Not found")

        if pm and wp:
            print(f"Phase Margin: {pm:.2f} deg at {wp:.2f} rad/s")
        else:
            print("Phase Margin: Not found")

        # Begin plotting
        plt.figure(figsize=(10, 8))

        # Magnitude plot
        plt.subplot(3, 1, 1)
        plt.semilogx(omega, mag_db, label="Magnitude")
        plt.axhline(0, color="gray", linestyle="--")
        if wg:
            plt.axvline(wg, color="red", linestyle="--", label=f"GM at {wg:.2f} rad/s")
        plt.ylabel("Magnitude (dB)")
        plt.title("Open-Loop Bode Plot (Filtered PID Controller)")
        plt.grid(True)
        plt.legend()

        # Phase plot
        plt.subplot(3, 1, 2)
        plt.semilogx(omega, phase_deg, label="Phase")
        plt.axhline(-180, color="gray", linestyle="--")
        if wp:
            plt.axvline(wp, color="green", linestyle="--", label=f"PM at {wp:.2f} rad/s")
        plt.ylabel("Phase (deg)")
        plt.xlabel("Frequency (rad/s)")
        plt.grid(True)
        plt.legend()

        # Step response plot
        t, y = self.compute_step_response()
        plt.subplot(3, 1, 3)
        plt.plot(t, y)
        plt.ylabel("Output")
        plt.xlabel("Time (s)")
        plt.grid(True)
        plt.title("Closed-Loop Step Response")

        plt.tight_layout()
        plt.show()
