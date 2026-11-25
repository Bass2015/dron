import pybullet as p
import numpy as np

from config import *
from general_helpers import flatten_tuple


class Lidar:
    UP_ID = 0
    FRONT_ID = 1
    DOWN_ID = -1

    def __init__(self, max_distance, n_side_sensors) -> None:
        self.max_distance = max_distance
        self.n_side_sensors = n_side_sensors
        self.n_sensors = n_side_sensors + 2
        self.sensor_vector = self.__compute_sensor_vector()

    def hit_distances(self, from_position: np.ndarray):
        """
        Computes the hit statuses and distances for multiple drones, given a starting position and a sensor configuration.

        Parameters:
            from_position : np.ndarray
                A 2D numpy array of shape `(n_drones, 3)` representing the starting positions of `n_drones`. Each position is a
                3D vector `[x, y, z]` indicating the coordinates of a drone in space.

        Returns:
            hits : np.ndarray
                A 2D boolean numpy array of shape `(n_drones, n_sensors)` where each entry is `True` if a sensor registered a hit
                and `False` otherwise.

            distances : np.ndarray
                A 2D numpy array of shape `(n_drones, n_sensors)` where each entry represents the distance to the nearest detected
                obstacle along each sensor's ray. If a sensor did not hit an obstacle, the distance value will be `-1`.
        """
        n_drones = from_position.shape[0]
        from_position = np.repeat(from_position, repeats=self.n_sensors, axis=0)
        add_vector = np.tile(self.sensor_vector, reps=(n_drones, 1))
        to_position = from_position + add_vector
        ray_casts = p.rayTestBatch(
            from_position,
            to_position,
        )
        results = np.asarray(list(flatten_tuple(ray_casts)))
        results = results.reshape((self.n_sensors * n_drones, RAY_TEST_FIELDS))
        hits = results[:, HIT_ID] != -1
        distances = results[:, HIT_FRACTION_ID] * self.max_distance
        distances[~hits] = -1
        return hits.reshape(n_drones, self.n_sensors), distances.reshape(
            n_drones, self.n_sensors
        )

    def __compute_sensor_vector(self):
        vector = np.zeros((self.n_sensors, 3))
        vector[[Lidar.UP_ID, Lidar.DOWN_ID], Z] = self.max_distance, -self.max_distance
        angles = np.linspace(0, np.pi * 2, self.n_side_sensors + 1)[:-1]
        vector[1:-1, X] = self.max_distance * np.cos(angles)
        vector[1:-1, Y] = self.max_distance * np.sin(angles)
        return vector
