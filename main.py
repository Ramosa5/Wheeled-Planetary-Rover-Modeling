import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants
g = 1.62  # Moon gravity [m/s^2]

# Body parameters
m_body = 8000  # kg
Jx, Jy, Jz = 10000, 70000, 72000  # kg*m^2
J_body = np.diag([Jx, Jy, Jz])  # inertia matrix

# Suspension parameters (8 wheels)
n_wheels = 8
m_wheel = 120  # kg
k_spring = 60000  # N/m
c_damper = 4600  # Ns/m
spring_rest_length = 0.9259  # m

# Wheel-ground interaction
k_wheel = 157000  # N/m
c_wheel = 4500    # Ns/m

# Wheel contact model
wheel_radius = 0.8       # m
kw = 157000              # N/m (vertical stiffness)
cw = 4500                # Ns/m (vertical damping)
cd = 50000               # N/rad (cornering stiffness)
mu_x_max = 0.6           # longitudinal traction limit
mu_y_max = 0.6           # lateral traction limit

# Motor torque limits
motor_max_torque = 700  # Nm

# Generalized coordinates: 6 DOF for body + 8 suspension springs
n_dof = 6 + n_wheels

# Initial state: flat start, zero velocity
x0 = np.zeros(n_dof)       # positions
x_dot0 = np.zeros(n_dof)   # velocities
initial_state = np.concatenate((x0, x_dot0))

def terrain_height(x, y):
    # Flat ground for now, but you could add a random profile
    return 0.0

def compute_wheel_forces(q, q_dot, i, wheel_pos_body):
    # Get global rotation
    yaw, roll, pitch = q[3:6]
    R = rotation_matrix(yaw, roll, pitch)

    # Global position of wheel center
    body_pos = q[0:3]
    suspension_deflection = q[6 + i]
    wheel_local = wheel_pos_body[i] + np.array([0, 0, -suspension_deflection])
    wheel_global = body_pos + R @ wheel_local

    # Terrain contact (assume terrain normal is [0, 0, 1])
    terrain_z = terrain_height(wheel_global[0], wheel_global[1])
    penetration = terrain_z + wheel_radius - wheel_global[2]
    penetration = max(penetration, 0.0)

    # Vertical velocity (approximate)
    v_global = q_dot[0:3] + np.cross(q_dot[3:6], R @ wheel_local)
    vertical_velocity = -v_global[2]

    # Normal force
    Fn = kw * penetration + cw * vertical_velocity if penetration > 0 else 0.0

    # Driving force (from torque)
    Mw = 0  # Placeholder — you'll define control later
    Fx = Mw / wheel_radius

    # Lateral (cornering) force
    Fy = 0  # To compute α (sideslip) you'd need wheel velocity vector

    return Fn, Fx, Fy


def rotation_matrix(yaw, roll, pitch):
    # R1: yaw (ψ)
    R1 = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1]
    ])
    # R2: roll (ϕ)
    R2 = np.array([
        [1, 0,             0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    # R3: pitch (θ)
    R3 = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0,             1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    return R1 @ R2 @ R3

def angular_velocity_matrix(yaw, roll, pitch):
    # Create transformation matrix T for angular velocity ω = T * [ψ_dot, ϕ_dot, θ_dot]
    T = np.array([
        [0, -np.sin(roll), np.cos(roll)*np.cos(pitch)],
        [0,  np.cos(roll), np.sin(roll)*np.cos(pitch)],
        [1, 0, -np.sin(pitch)]
    ])
    return T


def rover_dynamics(t, state):
    q = state[:n_dof]
    q_dot = state[n_dof:]

    # Extract positions and angles
    x, y, z = q[0:3]
    yaw, roll, pitch = q[3:6]
    x_dot, y_dot, z_dot = q_dot[0:3]
    yaw_dot, roll_dot, pitch_dot = q_dot[3:6]

    # Rotation matrix from body to inertial frame
    R = rotation_matrix(yaw, roll, pitch)

    # Angular velocity in body frame
    T_ang = angular_velocity_matrix(yaw, roll, pitch)
    omega_body = T_ang @ q_dot[3:6]

    # Initialize mass matrix
    M = np.eye(n_dof)
    M[0:3, 0:3] *= m_body
    M[3:6, 3:6] = J_body

    # Hard-coded wheel positions (from paper Table I)
    wheel_pos_body = [
        np.array([4.6, 2.905, -0.4894]),
        np.array([4.6, -2.905, -0.4894]),
        np.array([1.4, 2.905, -0.4894]),
        np.array([1.4, -2.905, -0.4894]),
        np.array([-2.2, 2.905, -0.4894]),
        np.array([-2.2, -2.905, -0.4894]),
        np.array([-5.4, 2.905, -0.4894]),
        np.array([-5.4, -2.905, -0.4894]),
    ]

    for i in range(n_wheels):
        M[6+i, 6+i] = m_wheel

    # Generalized forces
    F = np.zeros(n_dof)

    # Gravity
    F[2] -= m_body * g

    # Suspension spring/damper + gravity
    # Contact forces for each wheel
    for i in range(n_wheels):
        Fn, Fx, Fy = compute_wheel_forces(q, q_dot, i, wheel_pos_body)

        # Approximate contributions (you can make this more precise)
        # Body vertical force from wheels
        F[2] += Fn

        # Longitudinal and lateral forces — ideally added via Jacobian mapping
        # Here, simplified as body frame forces
        F[0] += Fx * np.cos(q[3])  # Apply forward force in x-direction (approx.)
        F[1] += Fy * np.sin(q[3])  # Apply lateral force in y-direction (approx.)

        disp = q[6 + i] - spring_rest_length
        vel = q_dot[6 + i]
        F[6 + i] = -k_spring * disp - c_damper * vel - m_wheel * g

    # Solve M * q_ddot = F
    q_ddot = np.linalg.solve(M, F)

    return np.concatenate((q_dot, q_ddot))


t_span = (0, 20)  # Simulate for 20 seconds
t_eval = np.linspace(t_span[0], t_span[1], 1000)

sol = solve_ivp(rover_dynamics, t_span, initial_state, t_eval=t_eval, method='RK45')

# Extract position and orientation
q_sol = sol.y[:n_dof, :]
time = sol.t

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time, q_sol[0], label='x')
plt.plot(time, q_sol[1], label='y')
plt.plot(time, q_sol[2], label='z')
plt.title('Rover Center of Mass Position')
plt.ylabel('Position [m]')
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(time, np.rad2deg(q_sol[3]), label='Yaw [°]')
plt.plot(time, np.rad2deg(q_sol[4]), label='Roll [°]')
plt.plot(time, np.rad2deg(q_sol[5]), label='Pitch [°]')
plt.ylabel('Orientation [deg]')
plt.xlabel('Time [s]')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
