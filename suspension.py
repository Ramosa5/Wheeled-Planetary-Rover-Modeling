class Suspension:
    def __init__(self, stiffness, damping):
        self.stiffness = stiffness
        self.damping = damping

    def calculateForce(self, disp, vel):
        return -self.stiffness * disp - self.damping * vel