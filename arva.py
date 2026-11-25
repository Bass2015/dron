import numpy as np
import pybullet as p
from config import *


def transformation_matrix_4d(positions: np.array):
    """
    Compute a transformation matrix for each 3D position in a set of input positions.

    This function generates a 3x3 transformation matrix for each point in a given
    array of 3D positions. The matrix is calculated based on specific mathematical
    relationships involving the x, y, and z coordinates. The resulting matrix set can
    be useful for transformations or rotations in 3D space.

    Parameters:
    ----------
    positions : np.ndarray
        An array of shape (N, 3) where each row represents the coordinates (x, y, z)
        of a point in 3D space.

    Returns:
    -------
    np.ndarray
        A 3-dimensional array of shape (N, 3, 3) where each 3x3 matrix corresponds
        to a transformation matrix for each input position.

    Example:
    --------
    >>> positions = np.array([[1, 2, 3], [4, 5, 6]])
    >>> transformation_matrix_4d(positions)
    array([[[ -7.,   6.,   9.],
            [  6., -11.,  18.],
            [  9.,  18., -13.]],

           [[ 23.,  60.,  72.],
            [ 60.,   7.,  90.],
            [ 72.,  90.,  41.]]])

    Notes:
    ------
    The transformation matrix for each point is calculated as follows:
      - Diagonal elements:
          matrix[i, 0, 0] = 2 * x^2 - y^2 - z^2
          matrix[i, 1, 1] = 2 * y^2 - x^2 - z^2
          matrix[i, 2, 2] = 2 * z^2 - x^2 - y^2
      - Off-diagonal elements:
          matrix[i, 0, 1] = matrix[i, 1, 0] = 3 * x * y
          matrix[i, 0, 2] = matrix[i, 2, 0] = 3 * x * z
          matrix[i, 1, 2] = matrix[i, 2, 1] = 3 * y * z
    """
    matrix = np.zeros((positions.shape[0], 3, 3))
    x, y, z = (
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
    )
    x2, y2, z2 = x**2, y**2, z**2
    matrix[:, 0, 0] = 2 * x2 - y2 - z2
    matrix[:, 1, 1] = 2 * y2 - x2 - z2
    matrix[:, 2, 2] = 2 * z2 - y2 - x2
    matrix[:, 0, 1] = matrix[:, 1, 0] = 3 * x * y
    matrix[:, 0, 2] = matrix[:, 2, 0] = 3 * x * z
    matrix[:, 1, 2] = matrix[:, 2, 1] = 3 * y * z
    return matrix


class ARVATransmitter:
    def __init__(
        self,
        position: np.array,
        rotation: np.array,
        intensity_scaler: float = 1.0,
    ) -> None:
        self.position = position
        self.rotation = rotation
        self.intensity_scaler = intensity_scaler

    def reset(self):
        self.position[[X, Y]] = np.random.uniform(
            low=-MAX_DISTANCE, high=MAX_DISTANCE, size=2
        )

    def vector_field(self, receiver_positions: np.array):
        """
        Calculate the vector field at given receiver positions relative to the object's position,
        following [Piniés and Tardós](https://ieeexplore.ieee.org/document/1642314).

        This function computes a vector field by applying a transformation matrix and rotation
        to the vector between the receiver positions and the object's position. It normalizes
        the result by a scaling factor based on the distance and returns the normalized vector field.

        Parameters:
        ----------
        receiver_positions : np.ndarray
            An array of shape (N, 3) containing the x, y, z coordinates of the receiver positions.

        Returns:
        -------
        np.ndarray
            A normalized array representing the vector field at each receiver position.
        """
        r_t_vector = receiver_positions - self.position
        numerator = transformation_matrix_4d(r_t_vector) @ self.rotation
        denominator = 4 * np.pi * np.linalg.norm(r_t_vector) ** 5
        result = numerator / denominator
        result = result / np.linalg.norm(result)
        return result

    def signal_intensity(self, receiver_positions: np.array):
        """
        Compute the signal intensity at specified receiver positions based on the transmitter's rotation
        and position.

        This function calculates the intensity of the signal received at each position by taking into
        account the object's orientation and relative position. It normalizes the signal by distance
        and applies a scaling factor for the intensity.

        Parameters:
        ----------
        receiver_positions : np.ndarray
            An array of shape (N, 3) with x, y, z coordinates of the receiver positions.

        Returns:
        -------
        np.ndarray
            An array of signal intensity values for each receiver position.
        """
        E1 = np.array([[1, 0, 0]]).T
        M = self.rotation.T @ E1 * E1.T @ self.rotation
        r_t_vector = receiver_positions - self.position
        r_t_norm = np.linalg.norm(r_t_vector, axis=1)
        inside_sqrt = 1 + (3 * np.diag(r_t_vector * M @ r_t_vector.T)) / r_t_norm**2
        intensity = (self.intensity_scaler * inside_sqrt**0.5) / (
            4 * np.pi * r_t_norm**3
        )
        return intensity

    def show_local_axes(self, client):
        AXIS_LENGTH = 1
        p.addUserDebugLine(
            lineFromXYZ=[0, 0, 0],
            lineToXYZ=[AXIS_LENGTH, 0, 0],
            lineColorRGB=[1, 0, 0],
            physicsClientId=client,
        )
        p.addUserDebugLine(
            lineFromXYZ=[0, 0, 0],
            lineToXYZ=[0, AXIS_LENGTH, 0],
            lineColorRGB=[1, 0, 0],
            physicsClientId=client,
        )
        p.addUserDebugLine(
            lineFromXYZ=[0, 0, 0],
            lineToXYZ=[0, 0, AXIS_LENGTH],
            lineColorRGB=[1, 0, 0],
            physicsClientId=client,
        )
