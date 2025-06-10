import numpy as np

class ExponentialSchedule:
    def __init__(self, epsilon_max=1.0, epsilon_min=0.1, decay=50000):
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.decay = decay

    def __call__(self, num_timesteps):
        eps = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-num_timesteps / self.decay)
        return max(self.epsilon_min, eps)