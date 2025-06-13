import numpy as np


class Controller:
    def __init__(self, v_target, kp, kd, M_sat):
        self.v_target = v_target
        self.kp = kp
        self.kd = kd
        self.M_sat = M_sat
        self.yaw_target = 0.0

    def calculateControlInput(self, vx, vy, yaw):
        v_mag = np.hypot(vx, vy)
        M0 = np.clip(self.kp * (self.v_target - v_mag), -self.M_sat, self.M_sat)
        yaw_err = (self.yaw_target - yaw + np.pi) % (2*np.pi) - np.pi
        dM = np.clip(self.kd * yaw_err, -self.M_sat, self.M_sat)
        return np.array([M0 + dM, M0 - dM, M0 + dM, M0 - dM])


