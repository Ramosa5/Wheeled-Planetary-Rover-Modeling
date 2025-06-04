class Wheel:
    def __init__(self, radius, torque=0.0):
        self.radius = radius
        self.torque = torque

    def applyTorque(self, torque):
        self.torque = torque