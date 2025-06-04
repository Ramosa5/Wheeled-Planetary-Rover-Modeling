import numpy as np


class TerrainModel:
    def getSurfaceHeight(self, x, y):
        slope_deg = 10
        base_height = np.tan(np.radians(slope_deg)) * x
        noise = np.sin(0.5 * x) + 0.5 * np.cos(0.3 * y)
        return base_height + 0.2 * noise

    def getSurfaceSlope(self, x, y):
        dx = 1e-3
        dy = 1e-3
        dzdx = (self.getSurfaceHeight(x + dx, y) - self.getSurfaceHeight(x, y)) / dx
        dzdy = (self.getSurfaceHeight(x, y + dy) - self.getSurfaceHeight(x, y)) / dy
        return dzdx, dzdy
