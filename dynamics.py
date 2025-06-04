import numpy as np

class DynamicModel:
    def __init__(self, terrain, n_dof, m_body, g, n_wheels, wheel_pos_body, wheel_radius, kw, cw, cd, mu_x_max, mu_y_max, spring_rest_length, m_wheel, J_body):
        self.terrain = terrain
        self.n_dof = n_dof
        self.m_body = m_body
        self.g = g
        self.n_wheels = n_wheels
        self.wheel_pos_body = wheel_pos_body
        self.wheel_radius = wheel_radius
        self.kw = kw
        self.cw = cw
        self.cd = cd
        self.mu_x_max = mu_x_max
        self.mu_y_max = mu_y_max
        self.spring_rest_length = spring_rest_length
        self.m_wheel = m_wheel
        self.J_body = J_body
    def rotation_matrix(self, yaw, roll, pitch):
        R1 = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])
        R2 = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
        R3 = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
        return R1 @ R2 @ R3

    def body_frame_velocity(self, q_dot, yaw, roll, pitch):
        R = self.rotation_matrix(yaw, roll, pitch)
        v_inertial = q_dot[0:3]
        v_body = R.T @ v_inertial
        return v_body[0]


    def skew(self, v):
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])
    def computeForces(self, q, q_dot, wheels, suspension, controller):
        F = np.zeros(self.n_dof)
        F[2] -= self.m_body * self.g
        v_x = self.body_frame_velocity(q_dot, q[3], q[4], q[5])
        M_left, M_right = controller.calculateControlInput(v_x)
        F_gen = np.zeros(self.n_dof)

        for i in range(self.n_wheels):
            Mw = M_left if i % 2 == 0 else M_right
            yaw, roll, pitch = q[3:6]
            R = self.rotation_matrix(yaw, roll, pitch)
            suspension_deflection = q[6 + i]
            wheel_local = self.wheel_pos_body[i] + np.array([0, 0, -suspension_deflection])
            wheel_global = q[0:3] + R @ wheel_local
            terrain_z = self.terrain.getSurfaceHeight(wheel_global[0], wheel_global[1])
            penetration = max(terrain_z + self.wheel_radius - wheel_global[2], 0.0)
            v_global = q_dot[0:3] + np.cross(q_dot[3:6], R @ wheel_local)
            v_wheel = R.T @ v_global
            vertical_velocity = -v_global[2]
            Fn = self.kw * penetration + self.cw * vertical_velocity if penetration > 0 else 0.0
            Fx = Mw / self.wheel_radius
            eps = 1e-2
            slip_angle = np.arctan2(v_wheel[1], abs(v_wheel[0]) + eps)
            Fy = np.clip(-self.cd * slip_angle, -self.mu_y_max * Fn, self.mu_y_max * Fn)
            F_wheel_local = np.array([Fx, Fy, Fn])
            J = np.zeros((self.n_dof, 3))
            r = R @ wheel_local
            J[0:3, 0:3] = np.eye(3)
            J[3:6, 0:3] = self.skew(r)
            J[6 + i, 2] = -1
            F_gen += J @ F_wheel_local
            disp = q[6 + i] - self.spring_rest_length
            vel = q_dot[6 + i]
            F_gen[6 + i] += suspension.calculateForce(disp, vel) - self.m_wheel * self.g

        M = np.eye(self.n_dof)
        M[0:3, 0:3] *= self.m_body
        M[3:6, 3:6] = self.J_body
        for i in range(self.n_wheels):
            M[6 + i, 6 + i] = self.m_wheel

        q_ddot = np.linalg.solve(M, F + F_gen)
        return np.concatenate((q_dot, q_ddot))
