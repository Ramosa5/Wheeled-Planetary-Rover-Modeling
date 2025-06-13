import numpy as np
class DynamicModel:
    def __init__(self, m_body, g, Jz, Jx, Jy, c_r, c_p, c_q):
        self.m_body = m_body
        self.g = g
        self.Jz, self.Jx, self.Jy = Jz, Jx, Jy
        self.c_r, self.c_p, self.c_q = c_r, c_p, c_q
        self.x = self.y = self.yaw = 0.0
        self.vx = self.vy = self.r = 0.0
        self.roll = self.pitch = 0.0
        self.p = self.q = 0.0
        self.z = 1.4153
        self.vz = 0.0

    def update(self, terrain_forces, suspension_forces, config, dt):
        # Vertical motion
        F_springs = suspension_forces.sum()
        F_total = F_springs - self.m_body * self.g
        az = F_total / self.m_body
        self.vz += az * dt
        self.z += self.vz * dt

        # Longitudinal and lateral
        F_long_tot = terrain_forces['long'].sum()
        F_lat_tot = terrain_forces['lat'].sum()
        Mz = (np.dot(config['xy_pos'][:,0], terrain_forces['lat'])
              - np.dot(config['xy_pos'][:,1], terrain_forces['long'])
              - self.c_r * self.r)
        # correct total mass
        m_tot = self.m_body + 4 * config['m_wheel']
        ax = F_long_tot / m_tot
        ay = F_lat_tot / m_tot

        self.vx += ax * dt
        self.vy += ay * dt
        self.x += self.vx * dt
        self.y += self.vy * dt

        self.r += (Mz / self.Jz) * dt
        self.yaw += self.r * dt

        Mx = np.dot(config['xy_pos'][:,1], suspension_forces) - self.c_p * self.p
        My = np.dot(config['xy_pos'][:,0], suspension_forces) - self.c_q * self.q
        self.p += (Mx / self.Jx) * dt
        self.q += (My / self.Jy) * dt
        self.roll += self.p * dt
        self.pitch += self.q * dt
