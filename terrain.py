import numpy as np

class TerrainModel:
    def __init__(self, amplitude=0.2, freq_x=0.2, freq_y=0.1):
        self.amplitude = amplitude
        self.freq_x = freq_x
        self.freq_y = freq_y

    def getSurfaceHeight(self, x, y):
        return self.amplitude * np.sin(self.freq_x * x) * np.cos(self.freq_y * y)
