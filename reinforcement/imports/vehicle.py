import numpy as np


class Vehicle:
    def __init__(self, mean_initial, var_initial):
        self.position: np.ndarray = var_initial * np.random.randn(2) + mean_initial  # [x, y]
        self.last_direction: list = [0, 0]

    def grid_move(self, step_size):
        directions = [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]]
        if any(self.last_direction):  # was last moving, last_direction != [0, 0]
            if np.random.rand() < 0.98:  # % chance to keep direction
                self.position += np.multiply(step_size, self.last_direction)
                return
            else:
                directions.remove([coordinate * -1 for coordinate in self.last_direction])  # no u turn

        directions.remove(self.last_direction)  # dont keep direction

        self.last_direction = directions[np.random.choice(len(directions))]  # roll from remaining directions
        self.position += np.multiply(step_size, self.last_direction)  # add step to position
