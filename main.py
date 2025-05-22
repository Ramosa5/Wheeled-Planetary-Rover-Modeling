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

# Torque control parameters
k_p = 1500        # Proportional gain
v_target = 0.5    # m/s desired forward speed
delta_M = 300     # Nm differential torque

# Wheel contact model
wheel_radius = 0.8       # m
kw = 157000              # N/m (vertical stiffness)
cw = 4500                # Ns/m (vertical damping)
cd = 50000               # N/rad (cornering stiffness)
mu_x_max = 0.6           # longitudinal traction limit
mu_y_max = 0.6           # lateral traction limit

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

# Motor torque limits
motor_max_torque = 700  # Nm

# Generalized coordinates: 6 DOF for body + 8 suspension springs
n_dof = 6 + n_wheels

# Initial state: flat start, zero velocity
x0 = np.zeros(n_dof)       # positions
x_dot0 = np.zeros(n_dof)   # velocities
initial_state = np.concatenate((x0, x_dot0))

def terrain_height(x, y):
    slope_deg = 10                      # 10 degree slope
    roughness_amplitude = 0.2          # height variation [m]
    roughness_scale_x = 0.5            # frequency along x
    roughness_scale_y = 0.3            # frequency along y

    # Global slope
    base_height = np.tan(np.radians(slope_deg)) * x

    # Bumpy terrain
    noise = (
        np.sin(roughness_scale_x * x) +
        0.5 * np.cos(roughness_scale_y * y)
    )

    return base_height + roughness_amplitude * noise

def compute_jacobian(q, wheel_local, i):
    """
    Returns a (n_dof x 3) Jacobian mapping contact force to generalized forces.
    """
    J = np.zeros((n_dof, 3))

    # Rotation matrix
    yaw, roll, pitch = q[3:6]
    R = rotation_matrix(yaw, roll, pitch)

    # Torque = r x F -> orientation torque contribution
    r = R @ wheel_local

    J[0:3, 0:3] = np.eye(3)        # Force maps directly to linear motion
    J[3:6, 0:3] = skew(r)          # Torque due to moment arm
    J[6 + i, 2] = -1               # Vertical (Z) force on suspension

    return J

def skew(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def body_frame_velocity(q_dot, yaw, roll, pitch):
    R = rotation_matrix(yaw, roll, pitch)
    v_inertial = q_dot[0:3]
    v_body = R.T @ v_inertial  # Transform to body frame
    return v_body[0]  # Forward velocity

def compute_wheel_forces(q, q_dot, i, wheel_pos_body, Mw):
    yaw, roll, pitch = q[3:6]
    R = rotation_matrix(yaw, roll, pitch)

    # Wheel local and global position
    suspension_deflection = q[6 + i]
    wheel_local = wheel_pos_body[i] + np.array([0, 0, -suspension_deflection])
    wheel_global = q[0:3] + R @ wheel_local

    # Terrain contact
    terrain_z = terrain_height(wheel_global[0], wheel_global[1])
    penetration = max(terrain_z + wheel_radius - wheel_global[2], 0.0)

    # Linear velocity at wheel
    v_global = q_dot[0:3] + np.cross(q_dot[3:6], R @ wheel_local)
    v_wheel = R.T @ v_global  # Transform to body frame

    vertical_velocity = -v_global[2]
    Fn = kw * penetration + cw * vertical_velocity if penetration > 0 else 0.0

    # Longitudinal (drive) force
    Fx = Mw / wheel_radius

    # --- NEW: Lateral (cornering) force ---
    eps = 1e-2
    slip_angle = np.arctan2(v_wheel[1], abs(v_wheel[0]) + eps)
    Fy_raw = -cd * slip_angle

    # Friction limit
    Fy_max = mu_y_max * Fn
    Fy = np.clip(Fy_raw, -Fy_max, Fy_max)

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

    for i in range(n_wheels):
        M[6+i, 6+i] = m_wheel

    # Generalized forces
    F = np.zeros(n_dof)

    # Gravity
    F[2] -= m_body * g

    # Longitudinal speed in body frame
    v_x = body_frame_velocity(q_dot, q[3], q[4], q[5])

    # Average torque control
    M0 = k_p * (v_target - v_x)

    # Distribute to left/right wheels
    M_left = np.clip(M0 + delta_M / 2, -motor_max_torque, motor_max_torque)
    M_right = np.clip(M0 - delta_M / 2, -motor_max_torque, motor_max_torque)

    # Suspension spring/damper + gravity
    # Contact forces for each wheel
    F_gen = np.zeros(n_dof)

    for i in range(n_wheels):
        Mw = M_left if i % 2 == 0 else M_right
        Fn, Fx, Fy = compute_wheel_forces(q, q_dot, i, wheel_pos_body, Mw)

        # Local contact force in body frame
        F_wheel_local = np.array([Fx, Fy, Fn])  # [longitudinal, lateral, normal]

        # Compute local wheel position (incl. suspension)
        suspension_deflection = q[6 + i]
        wheel_local = wheel_pos_body[i] + np.array([0, 0, -suspension_deflection])

        # Jacobian projection
        J_i = compute_jacobian(q, wheel_local, i)
        F_gen += J_i @ F_wheel_local

        # Add suspension spring-damper forces directly
        disp = q[6 + i] - spring_rest_length
        vel = q_dot[6 + i]
        F_gen[6 + i] += -k_spring * disp - c_damper * vel - m_wheel * g

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

v_x_history = [body_frame_velocity(sol.y[n_dof:,i], *sol.y[3:6, i]) for i in range(len(sol.t))]

plt.figure()
plt.plot(sol.t, v_x_history)
plt.axhline(v_target, color='r', linestyle='--', label='Target')
plt.xlabel('Time [s]')
plt.ylabel('Forward Velocity [m/s]')
plt.title('Rover Longitudinal Speed')
plt.legend()
plt.grid()
plt.show()

lateral_forces = []

for i in range(len(sol.t)):
    q_i = sol.y[:n_dof, i]
    q_dot_i = sol.y[n_dof:, i]

    yaw, roll, pitch = q_i[3:6]
    R = rotation_matrix(yaw, roll, pitch)
    v_global = q_dot_i[0:3] + np.cross(q_dot_i[3:6], R @ wheel_pos_body[0])
    v_wheel = R.T @ v_global
    slip_angle = np.arctan2(v_wheel[1], abs(v_wheel[0]) + 1e-2)
    Fy = -cd * slip_angle
    lateral_forces.append(np.clip(Fy, -mu_y_max * m_wheel * g, mu_y_max * m_wheel * g))

plt.figure()
plt.plot(sol.t, lateral_forces)
plt.title('Lateral Force on Front-Left Wheel')
plt.xlabel('Time [s]')
plt.ylabel('Force [N]')
plt.grid()
plt.show()

plt.figure()
plt.plot(sol.t, np.rad2deg(sol.y[3, :]))  # Yaw angle
plt.title('Rover Yaw Over Time')
plt.xlabel('Time [s]')
plt.ylabel('Yaw Angle [deg]')
plt.grid()
plt.show()

x_vals = np.linspace(0, 20, 200)
y_vals = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = terrain_height(X, Y)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='terrain', linewidth=0, antialiased=False)
ax.set_title('Terrain Height Map')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
plt.show()
