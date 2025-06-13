class Wheel:
    def __init__(self, radius: float, mass: float, damping_tau: float, J_w: float):
        self.radius = radius
        self.mass = mass
        self.damping_tau = damping_tau
        self.J_w = J_w
        self.omega = 0.0
        self.z = radius
        self.vz = 0.0

    def applyTorque(self, drive_torque, force_long, dt):
        tau_drive = drive_torque - self.radius * force_long
        omega_dot = (tau_drive - self.damping_tau * self.omega) / self.J_w
        self.omega += omega_dot * dt
