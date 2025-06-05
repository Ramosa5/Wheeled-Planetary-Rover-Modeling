import numpy as np
from scipy.integrate import solve_ivp

class Simulator:
    def __init__(self, rover, n_dof):
        self.rover = rover
        self.n_dof = n_dof
    def runSimulation(self):
        x0 = np.zeros(self.n_dof)
        x_dot0 = np.zeros(self.n_dof)
        state0 = np.concatenate((x0, x_dot0))
        t_span = (0, 20)
        t_eval = np.linspace(t_span[0], t_span[1], 10000)
        sol = solve_ivp(lambda t, y: self.rover.simulateStep(y, t), t_span, state0, t_eval=t_eval)
        return sol
