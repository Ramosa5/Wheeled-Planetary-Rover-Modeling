import numpy as np
from simulator import Simulator

if __name__ == "__main__":
    config = {
        'R_w': 0.80, 'm_wheel': 120.0, 'c_tau': 10.0, 'J_w': 12.0,
        'k': 60000.0, 'd': 4600.0, 'L0': 0.9259,
        'k_w': 157e3, 'c_w': 4.5e3,
        'm_body': 8000.0, 'g': 1.62,
        'Jz': 1.2e5, 'Jx': 1.0e4, 'Jy': 7.0e4, 'c_r': 1.0e4, 'c_p': 2.0e5, 'c_q': 2.0e5,
        'v_target': 5.0, 'kp': 3000.0, 'kd': 500.0, 'M_sat': 4000.0,
        'mu_x': 0.8, 'mu_y': 0.8, 'B_x': 10.0, 'C_x': 1.9, 'B_y': 5.0, 'C_y': 1.3,
        'dt': 0.001, 'T': 150.0,
        'xy_pos': np.array([[4.6, 2.905], [4.6, -2.905], [-2.2, 2.905], [-2.2, -2.905]])
    }
    sim = Simulator(config)
    hist = sim.runSimulation()
    sim.plotResults(hist)
