import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from wheel import Wheel
from suspension import Suspension
from terrain import TerrainModel
from sensor import Sensor
from controller import Controller
from dynamics import DynamicModel
from rover import Rover
from simulator import Simulator

# Constants
g = 1.62
n_wheels = 8
m_body = 8000
J_body = np.diag([10000, 70000, 72000])
m_wheel = 120
k_spring = 60000
c_damper = 4600
spring_rest_length = 0.9259
wheel_radius = 0.8
kw, cw, cd = 157000, 4500, 50000
mu_x_max, mu_y_max = 0.6, 0.6
k_p = 1500
v_target = 0.5
delta_M = 300
motor_max_torque = 700
n_dof = 6 + n_wheels

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

# Setup
wheels = [Wheel(radius=wheel_radius) for _ in range(n_wheels)]
suspension = Suspension(k_spring, c_damper)
controller = Controller(v_target, k_p, motor_max_torque, delta_M)
terrain_model = TerrainModel()
dynamic_model = DynamicModel(terrain_model, n_dof, m_body, g, n_wheels, wheel_pos_body, wheel_radius, kw, cw, cd, mu_x_max, mu_y_max, spring_rest_length, m_wheel, J_body)
sensors = [Sensor() for _ in range(4)]
rover = Rover(wheels, suspension, controller, dynamic_model, terrain_model, sensors, n_dof)
simulator = Simulator(rover, n_dof)

# Run simulation
sol = simulator.runSimulation()

# Plot Rover Position Over Time
plt.figure(figsize=(10, 4))
plt.plot(sol.t, sol.y[0], label='x position')
plt.plot(sol.t, sol.y[1], label='y position')
plt.plot(sol.t, sol.y[2], label='z position')
plt.xlabel('Time [s]')
plt.ylabel('Position [m]')
plt.title('Rover Position Over Time')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# 3D Trajectory Plot with Terrain
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Rover Trajectory Data
x_vals = sol.y[0]
y_vals = sol.y[1]
z_vals = sol.y[2]

# Plot Rover Trajectory
ax.plot(x_vals, y_vals, z_vals, label='Rover Trajectory', color='blue')

# Add terrain data
terrain_x = np.linspace(min(x_vals), max(x_vals), 100)
terrain_y = np.linspace(min(y_vals), max(y_vals), 100)
terrain_X, terrain_Y = np.meshgrid(terrain_x, terrain_y)

# Calculate the surface height of the terrain at each point
terrain_Z = terrain_model.getSurfaceHeight(terrain_X, terrain_Y)

# Plot the terrain surface
ax.plot_surface(terrain_X, terrain_Y, terrain_Z, cmap='YlGnBu', alpha=0.5, rstride=10, cstride=10)

# Set the labels and title
ax.set_title('Rover 3D Trajectory with Terrain')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.legend()
plt.tight_layout()
plt.show()

# Velocity Over Time
plt.figure(figsize=(10, 4))
velocity = np.linalg.norm(sol.y[3:6], axis=0)  # Prędkość całkowita (magnitude of velocity vector)
plt.plot(sol.t, velocity, label='Total velocity')
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')
plt.title('Rover Velocity Over Time')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Motor Torques Over Time (left and right wheels)
plt.figure(figsize=(10, 4))
M_left = np.clip(controller.calculateControlInput(sol.y[3])[0], -motor_max_torque, motor_max_torque)
M_right = np.clip(controller.calculateControlInput(sol.y[3])[1], -motor_max_torque, motor_max_torque)
plt.plot(sol.t, M_left, label='Left motor torque')
plt.plot(sol.t, M_right, label='Right motor torque')
plt.xlabel('Time [s]')
plt.ylabel('Torque [Nm]')
plt.title('Motor Torques Over Time')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Spring Displacements Over Time (showing displacement for each wheel)
plt.figure(figsize=(10, 4))
for i in range(n_wheels):
    spring_disp = sol.y[6 + i] - spring_rest_length
    plt.plot(sol.t, spring_disp, label=f'Spring {i+1} displacement')
plt.xlabel('Time [s]')
plt.ylabel('Spring Displacement [m]')
plt.title('Spring Displacements Over Time')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
