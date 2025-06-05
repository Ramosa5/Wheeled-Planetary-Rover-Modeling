import numpy as np


class TerrainModel:
    # def getSurfaceHeight(self, x, y):
    #     # Mniejsza zmienność terenu
    #     slope_deg = 10  # Kąt nachylenia
    #     base_height = np.tan(np.radians(slope_deg)) * x
    #
    #     # Zmniejszamy amplitudę hałasu
    #     noise = np.sin(0.1 * x) + 0.1 * np.cos(0.1 * y)
    #     return base_height + 0.05 * noise  # Zmniejszamy wpływ hałasu

    def getSurfaceSlope(self, x, y):
        dx = 1e-3
        dy = 1e-3

        dzdx = (self.getSurfaceHeight(x + dx, y) - self.getSurfaceHeight(x, y)) / dx
        dzdy = (self.getSurfaceHeight(x, y + dy) - self.getSurfaceHeight(x, y)) / dy
        return dzdx, dzdy

    def getSurfaceHeight(self, x, y):
        # Stała wysokość terenu (płaski teren)
        return -4.0  # Teren na wysokości 0 metrów (lub inna wartość, jeśli chcesz, by teren był na innym poziomie)
