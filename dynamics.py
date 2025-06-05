import numpy as np
import csv

class DynamicModel:
    def __init__(self, terrain, n_dof, m_body, g, n_wheels, wheel_pos_body, wheel_radius, kw, cw, cd, mu_x_max,
                 mu_y_max, spring_rest_length, m_wheel, J_body):
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

        # Initialize CSV logging
        self.log_file = 'forces_log.csv'
        self.init_csv_log()

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

    def computeMassMatrix(self, q):
        # Macierz masy ciała sztywnego
        M = np.eye(self.n_dof)
        M[0:3, 0:3] *= self.m_body
        M[3:6, 3:6] = self.J_body

        # Macierz masy dla kół (masy kół + ich wpływ na układ)
        for i in range(self.n_wheels):
            M[6 + i, 6 + i] = self.m_wheel

        # Dynamiczna zmiana masy w zależności od długości sprężyn
        for i in range(self.n_wheels):
            spring_disp = q[6 + i] - self.spring_rest_length  # Odchylenie sprężyny
            spring_mass_factor = 1 + 0.1 * abs(spring_disp)  # Prosty przykład
            M[6 + i, 6 + i] *= spring_mass_factor  # Zwiększanie masy kół zależnie od sprężyny

        return M

    def computeLagrangian(self, q, q_dot):
        # Obliczanie energii kinetycznej T
        T_translational = 0.5 * self.m_body * np.linalg.norm(q_dot[:3]) ** 2  # Energia translacyjna ciała sztywnego
        T_rotational = 0.5 * np.dot(q_dot[3:6], np.dot(self.J_body, q_dot[3:6]))  # Energia rotacyjna ciała sztywnego

        # Energia kinetyczna kół
        T_wheels = 0
        for i in range(self.n_wheels):
            wheel_velocity = q_dot[0:3] + np.cross(q_dot[3:6],
                                                   self.rotation_matrix(q[3], q[4], q[5]) @ self.wheel_pos_body[i])
            T_wheels += 0.5 * self.m_wheel * np.linalg.norm(wheel_velocity) ** 2

        # Całkowita energia kinetyczna
        T = T_translational + T_rotational + T_wheels

        # Obliczanie energii potencjalnej U
        U_gravity = self.m_body * self.g * q[2]  # Energia potencjalna z grawitacji
        U_spring = 0
        for i in range(self.n_wheels):
            spring_disp = q[6 + i] - self.spring_rest_length  # Odchylenie sprężyny
            U_spring += 0.5 * self.kw * spring_disp ** 2  # Energia potencjalna sprężyny

        # Całkowita energia potencjalna
        U = U_gravity + U_spring

        # Lagrangian
        L = T - U
        return L

    def computePartialLagrangian(self, q, q_dot, i, derivative="q"):
        epsilon = 1e-6  # Mała wartość do obliczeń różnic skończonych

        if derivative == "q_dot":
            # Pochodna względem prędkości (dot q_i)
            q_dot_plus = q_dot.copy()
            q_dot_minus = q_dot.copy()

            # Zwiększenie prędkości q_dot_i o epsilon
            q_dot_plus[i] += epsilon
            q_dot_minus[i] -= epsilon

            # Obliczanie Lagrangiana dla q_dot_i + epsilon i q_dot_i - epsilon
            L_plus = self.computeLagrangian(q, q_dot_plus)
            L_minus = self.computeLagrangian(q, q_dot_minus)

            # Pochodna numeryczna
            dL_dq_dot = (L_plus - L_minus) / (2 * epsilon)
            return dL_dq_dot

        elif derivative == "q":
            # Pochodna względem współrzędnych q_i
            q_plus = q.copy()
            q_minus = q.copy()

            # Zwiększenie współrzędnej q_i o epsilon
            q_plus[i] += epsilon
            q_minus[i] -= epsilon

            # Obliczanie Lagrangiana dla q_i + epsilon i q_i - epsilon
            L_plus = self.computeLagrangian(q_plus, q_dot)
            L_minus = self.computeLagrangian(q_minus, q_dot)

            # Pochodna numeryczna
            dL_dq = (L_plus - L_minus) / (2 * epsilon)
            return dL_dq
        return None

    def computeEquationsOfMotion(self, q, q_dot):
        # Obliczanie Lagrangianu
        L = self.computeLagrangian(q, q_dot)

        # Zastosowanie równań Lagrange'a
        q_ddot = np.zeros(self.n_dof)

        for i in range(self.n_dof):
            # Pochodne Lagrangiana względem prędkości i współrzędnych
            dL_dq_dot = self.computePartialLagrangian(q, q_dot, i, derivative="q_dot")
            dL_dq = self.computePartialLagrangian(q, q_dot, i, derivative="q")

            # Równania Lagrange'a
            q_ddot[i] = (dL_dq_dot - dL_dq)  # Po zastosowaniu równań Lagrange'a

        return q_ddot

    def init_csv_log(self):
        """Initialize CSV log file with headers."""
        with open(self.log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write header row
            writer.writerow(["Time", "F_gravity", "F_spring", "F_normal", "F_friction", "F_suspension"])

    def log_forces(self, time, F_gravity, F_spring, F_normal, F_friction, F_suspension):
        """Log forces to CSV."""
        with open(self.log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([time, F_gravity, F_spring, F_normal, F_friction, F_suspension])

    def computeForces(self, q, q_dot, wheels, suspension, controller, time):
        F = np.zeros(self.n_dof)
        F[2] -= self.m_body * self.g  # Gravity force on the rover

        # Initialize force counters
        total_F_spring = 0
        total_F_normal = 0
        total_F_friction = 0
        total_F_suspension = 0

        v_x = self.body_frame_velocity(q_dot, q[3], q[4], q[5])
        M_left, M_right = controller.calculateControlInput(v_x)
        F_gen = np.zeros(self.n_dof)

        # Mass matrix
        M = self.computeMassMatrix(q)

        # Loop over all wheels to compute forces
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

            total_F_normal += Fn  # Sum of normal forces
            F_x_max = self.mu_x_max * Fn
            Fx = np.clip(Mw / self.wheel_radius, -F_x_max, F_x_max)
            total_F_friction += Fx  # Sum of friction forces
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
            total_F_spring += suspension.calculateForce(disp, vel)  # Sum of suspension forces

        # Log the forces at each time step
        self.log_forces(time, F[2], total_F_spring, total_F_normal, total_F_friction, total_F_suspension)

        q_ddot = np.linalg.solve(M, F + F_gen)
        return np.concatenate((q_dot, q_ddot))