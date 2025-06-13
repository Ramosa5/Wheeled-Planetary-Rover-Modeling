import numpy as np
from wheel import Wheel
from suspension import Suspension
from dynamicmodel import DynamicModel
from controller import  Controller
from terrain import TerrainModel

class Rover:
    def __init__(self, config):
        self.config = config
        self.wheels = [Wheel(config['R_w'], config['m_wheel'], config['c_tau'], config['J_w']) for _ in range(4)]
        self.suspension = Suspension(config['k'], config['d'], config['L0'])
        self.dynamicModel = DynamicModel(
            config['m_body'], config['g'], config['Jz'], config['Jx'], config['Jy'],
            config['c_r'], config['c_p'], config['c_q'])
        self.controller = Controller(config['v_target'], config['kp'], config['kd'], config['M_sat'])
        self.terrainModel = TerrainModel()
        self.mu_x, self.mu_y = config['mu_x'], config['mu_y']
        self.B_x, self.C_x = config['B_x'], config['C_x']
        self.B_y, self.C_y = config['B_y'], config['C_y']

    def simulateStep(self, dt, t_i):
        # steering input
        M_cmd = self.controller.calculateControlInput(
            self.dynamicModel.vx, self.dynamicModel.vy, self.dynamicModel.yaw)
        # steering angle for front wheels
        if 30.0 < t_i < 35.0:
            alpha = np.array([0.1, 0.1, 0.0, 0.0])
        else:
            alpha = np.zeros(4)

        # wheel kinematics
        z = self.dynamicModel.z
        roll, pitch = self.dynamicModel.roll, self.dynamicModel.pitch
        z_off = np.array([-0.4894] * 4)
        z_mount = z + z_off + self.config['xy_pos'][:,0]*pitch + self.config['xy_pos'][:,1]*roll
        vz_mount = self.dynamicModel.vz + self.config['xy_pos'][:,0]*self.dynamicModel.q + self.config['xy_pos'][:,1]*self.dynamicModel.p

        # world positions
        cosy, siny = np.cos(self.dynamicModel.yaw), np.sin(self.dynamicModel.yaw)
        x_ws = self.dynamicModel.x + cosy*self.config['xy_pos'][:,0] - siny*self.config['xy_pos'][:,1]
        y_ws = self.dynamicModel.y + siny*self.config['xy_pos'][:,0] + cosy*self.config['xy_pos'][:,1]

        # terrain height
        z_ground = self.terrainModel.getSurfaceHeight(x_ws, y_ws)
        z_bottom = np.array([w.z for w in self.wheels]) - (z_ground + self.config['R_w'])
        F_n = np.where(z_bottom < 0,
                       -self.config['k_w']*z_bottom - self.config['c_w']*np.array([w.vz for w in self.wheels]),
                       0.0)

        # traction & tire model
        vxi = self.dynamicModel.vx - self.dynamicModel.r*self.config['xy_pos'][:,1]
        sigma = (self.config['R_w']*np.array([w.omega for w in self.wheels]) - vxi) / np.maximum(np.abs(vxi), 1e-3)
        D_x = self.mu_x * F_n
        D_y = self.mu_y * F_n
        F_long = D_x * np.sin(self.C_x * np.arctan(self.B_x * sigma))
        F_lat = D_y * np.sin(self.C_y * np.arctan(self.B_y * alpha))

        # wheel and suspension dynamics
        suspension_forces = np.zeros(4)
        for i, w in enumerate(self.wheels):
            suspension_forces[i] = self.suspension.calculateForce(z_mount[i] - w.z, vz_mount[i], w.vz)
            w.applyTorque(M_cmd[i], F_long[i], dt)
            az_w = (F_n[i] - suspension_forces[i] - self.config['m_wheel']*self.config['g']) / self.config['m_wheel']
            w.vz += az_w * dt
            w.z += w.vz * dt

        terrain_forces = {'long': F_long, 'lat': F_lat}
        self.dynamicModel.update(terrain_forces, suspension_forces, self.config, dt)
        self.last_susp_forces = suspension_forces
