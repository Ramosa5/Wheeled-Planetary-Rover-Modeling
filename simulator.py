import numpy as np
from rover import Rover
import matplotlib.pyplot as plt

class Simulator:
    def __init__(self, config):
        self.rover = Rover(config)
        self.dt = config['dt']
        self.T = config['T']
        self.steps = int(self.T / self.dt)

    def runSimulation(self):
        hist = {key: [] for key in ['x','y','z','yaw','vx','vy','roll','pitch','F_sum']}
        for wname in ['LF','RF','LR','RR']:
            hist[f'F_s_{wname}'] = []
        for i in range(self.steps):
            t_i = i * self.dt
            self.rover.simulateStep(self.dt, t_i)
            dm = self.rover.dynamicModel
            hist['x'].append(dm.x)
            hist['y'].append(dm.y)
            hist['z'].append(dm.z)
            hist['yaw'].append(np.rad2deg(dm.yaw))
            hist['vx'].append(dm.vx)
            hist['vy'].append(dm.vy)
            hist['roll'].append(np.rad2deg(dm.roll))
            hist['pitch'].append(np.rad2deg(dm.pitch))
            fs = self.rover.last_susp_forces
            hist['F_sum'].append(fs.sum())
            for idx, name in enumerate(['LF','RF','LR','RR']):
                hist[f'F_s_{name}'].append(fs[idx])
        return hist

    def plotResults(self, hist):
        t = np.linspace(0, self.T, self.steps)
        # 2D trajectory
        plt.figure(figsize=(5,5))
        plt.plot(hist['x'], hist['y'])
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title('Ślad CG po podłożu')
        plt.axis('equal')
        plt.grid(True)
        plt.show()
        # yaw + velocities
        fig, ax = plt.subplots(2,1, figsize=(10,6), sharex=True)
        ax[0].plot(t, hist['yaw']); ax[0].set_ylabel('yaw [°]'); ax[0].grid(True)
        ax[1].plot(t, hist['vx'], label='vx'); ax[1].plot(t, hist['vy'], label='vy')
        ax[1].set_xlabel('czas [s]'); ax[1].set_ylabel('prędkość [m/s]'); ax[1].legend(); ax[1].grid(True)
        plt.tight_layout(); plt.show()
        # linear speed
        speed = np.sqrt(np.array(hist['vx'])**2 + np.array(hist['vy'])**2)
        plt.figure(figsize=(10,4)); plt.plot(t, speed)
        plt.xlabel('czas [s]'); plt.ylabel('v [m/s]'); plt.title('Prędkość liniowa CG'); plt.grid(True); plt.show()
        # position over time
        plt.figure(figsize=(10,4))
        plt.plot(t, hist['x'], label='x [m]'); plt.plot(t, hist['y'], label='y [m]')
        plt.xlabel('czas [s]'); plt.ylabel('pozycja [m]'); plt.title('Pozycja CG w czasie'); plt.legend(); plt.grid(True); plt.show()
        # 3D trajectory
        fig=plt.figure(figsize=(8,6)); ax3=fig.add_subplot(111, projection='3d')
        ax3.plot(hist['x'], hist['y'], hist['z'], linewidth=2, label='traj CG')
        Xg, Yg = np.meshgrid(np.linspace(min(hist['x']), max(hist['x']), 50),
                             np.linspace(min(hist['y']), max(hist['y']), 50))
        Zg = self.rover.terrainModel.getSurfaceHeight(Xg, Yg)
        ax3.plot_surface(Xg, Yg, Zg, alpha=0.6, edgecolor='grey')
        step = max(1, len(hist['x'])//20)
        for i in range(0, len(hist['x']), step):
            dx, dy = np.cos(np.deg2rad(hist['yaw'][i])), np.sin(np.deg2rad(hist['yaw'][i]))
            ax3.quiver(hist['x'][i], hist['y'][i], hist['z'][i], dx, dy, 0, length=0.5)
        ax3.set_xlabel('X [m]'); ax3.set_ylabel('Y [m]'); ax3.set_zlabel('Z [m]')
        ax3.set_title('3D Trajektoria, teren i kierunek'); ax3.legend(); plt.tight_layout(); plt.show()
        # spring forces and body angles
        plt.figure(figsize=(10,4)); plt.plot(t, hist['F_sum']); plt.xlabel('czas [s]'); plt.ylabel('Σ F_s [N]');
        plt.title('Łączna siła sprężyn'); plt.grid(True); plt.show()
        plt.figure(figsize=(10,4)); plt.plot(t, hist['roll'], label='roll [°]'); plt.plot(t, hist['pitch'], label='pitch [°]');
        plt.xlabel('czas [s]'); plt.ylabel('kąt [°]'); plt.title('Przechyły roll/pitch'); plt.legend(); plt.grid(True); plt.show()
        plt.figure(figsize=(10,4)); plt.plot(t, hist['z']); plt.xlabel('czas [s]'); plt.ylabel('z CG [m]');
        plt.title('Ruch pionowy CG'); plt.grid(True); plt.show()
        plt.figure(figsize=(10,6));
        for name in ['LF','RF','LR','RR']:
            plt.plot(t, hist[f'F_s_{name}'], label=name)
        plt.xlabel('czas [s]'); plt.ylabel('F sprężyny [N]'); plt.title('Siły sprężyn'); plt.legend(); plt.grid(True); plt.show()
