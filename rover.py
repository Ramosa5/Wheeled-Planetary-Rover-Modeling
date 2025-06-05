
class Rover:
    def __init__(self, wheels, suspension, controller, dynamicModel, terrainModel, sensors, n_dof):
        self.wheels = wheels
        self.suspension = suspension
        self.controller = controller
        self.dynamicModel = dynamicModel
        self.terrainModel = terrainModel
        self.sensors = sensors
        self.n_dof = n_dof
    def simulateStep(self, state, t):
        q = state[:self.n_dof]
        q_dot = state[self.n_dof:]
        return self.dynamicModel.computeForces(q, q_dot, self.wheels, self.suspension, self.controller, t)
