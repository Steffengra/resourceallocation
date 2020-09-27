import unittest
import numpy as np

from reinforcement.imports.vehicle import Vehicle


class VehicleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.vehicle = Vehicle(100, 10)

    def test_grid_move(self):
        step = 1
        old_position = self.vehicle.position.copy()
        self.vehicle.grid_move(step_size=step)
        new_position = self.vehicle.position.copy()
        distance = np.sqrt(np.sum(np.power(old_position - new_position, 2)))

        self.assertEqual(step, distance)


if __name__ == '__main__':
    unittest.main()