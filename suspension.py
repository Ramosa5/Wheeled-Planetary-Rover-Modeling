import numpy as np
class Suspension:
    def __init__(self, k: float, d: float, L0: float):
        self.k = k
        self.d = d
        self.L0 = L0

    def calculateForce(self, l_spring, vz_mount, vz_wheel):
        delta = np.clip(self.L0 - l_spring, 0, None)
        F_s = self.k * delta
        F_d = -self.d * (vz_mount - vz_wheel)
        return F_s + F_d
