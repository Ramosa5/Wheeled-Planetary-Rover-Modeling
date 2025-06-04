import numpy as np

class Controller:
    def __init__(self, v_target, k_p, motor_max_torque, delta_M):
        self.v_target = v_target
        self.k_p = k_p
        self.motor_max_torque = motor_max_torque
        self.delta_M = delta_M
    def calculateControlInput(self, v_current):
        M0 = self.k_p * (self.v_target - v_current)
        M_left = np.clip(M0 + self.delta_M / 2, -self.motor_max_torque, self.motor_max_torque)
        M_right = np.clip(M0 - self.delta_M / 2, -self.motor_max_torque, self.motor_max_torque)
        return M_left, M_right