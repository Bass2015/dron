from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import (
    ActionType,
    DroneModel,
    ObservationType,
    Physics,
)
import numpy as np
import pybullet as p

from arva import ARVATransmitter
from config import *
from lidar import Lidar


class ARVADrone(BaseRLAviary):
    BASE_OBS_DIMENSIONS = 13
    N_ACTIONS = 4

    def __init__(
        self,
        lidar_range: int = 5,
        lidar_sensors: int = 6,
        drone_model: DroneModel = DroneModel.CF2X,
        num_drones: int = 1,
        neighbourhood_radius: float = np.inf,
        initial_xyzs=None,
        initial_rpys=None,
        arva_initial_xyz=np.zeros(3),
        arva_initial_rpy=np.repeat(np.nextafter(0, 1, dtype=np.float32), 3),
        physics: Physics = Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 240,
        obstacles=False,
        gui=False,
        user_debug_gui=False,
        record=False,
        obs: ObservationType = ObservationType.KIN,
        act: ActionType = ActionType.RPM,
    ):
        self.lidar = Lidar(max_distance=lidar_range, n_side_sensors=lidar_sensors)
        self.ACTION_BUFFER_SIZE = int(ctrl_freq // 2)
        self.obs_dimensions = (
            ARVADrone.BASE_OBS_DIMENSIONS
            # + self.lidar.n_sensors
            + ARVADrone.N_ACTIONS * self.ACTION_BUFFER_SIZE
        )

        self.arva_transmitter = ARVATransmitter(
            position=arva_initial_xyz,
            rotation=arva_initial_rpy,
        )
        super().__init__(
            drone_model,
            num_drones,
            neighbourhood_radius,
            initial_xyzs,
            initial_rpys,
            physics,
            pyb_freq,
            ctrl_freq,
            obstacles,
            gui,
            record,
            user_debug_gui,
            obs=ObservationType.KIN,
            act=ActionType.RPM,
        )

    def _housekeeping(self):
        super()._housekeeping()
        self.pos[:, [X, Y]] = np.random.uniform(
            low=-MAX_DISTANCE, high=MAX_DISTANCE, size=(self.NUM_DRONES, 2)
        )
        self.arva_transmitter.reset()
        self.arva_transmitter.show_local_axes(self.CLIENT)
        p.loadURDF(
            "duck_vhacd.urdf",
            self.arva_transmitter.position,
            p.getQuaternionFromEuler(self.arva_transmitter.rotation),
            physicsClientId=self.CLIENT,
        )

    def step(self, action):
        return super().step(action)

    def _observationSpace(self):
        """Returns the observation space of the environment.

        OBSERVATION_IDX:
            pos: [0, 1, 2],
            rot: [3, 4, 5],
            vel: [6, 7, 8],
            ang_vel: [9, 10, 11],
            intensity: 12,
            lidar: [13, 13+n_lidar_sensors],
            action_buffer = [-actions*action_buffer_size:]

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,13 + n_lidar_sensors + n_acctions * action_buffer_size) depending on the observation type.

        """

        lo = -np.inf
        hi = np.inf

        obs_lower_bound, obs_upper_bound = (
            np.zeros((self.NUM_DRONES, self.obs_dimensions)),
            np.zeros((self.NUM_DRONES, self.obs_dimensions)),
        )

        obs_lower_bound[:, : OBSERVATION_IDX["intensity"]] = lo
        obs_lower_bound[:, OBSERVATION_IDX["pos"][Z]] = 0
        obs_lower_bound[:, -ARVADrone.N_ACTIONS * self.ACTION_BUFFER_SIZE :] = -1

        obs_upper_bound[:, : ARVADrone.BASE_OBS_DIMENSIONS] = hi
        # obs_upper_bound[
        #     :,
        #     ARVADrone.BASE_OBS_DIMENSIONS : ARVADrone.BASE_OBS_DIMENSIONS
        #     + self.lidar.n_sensors,
        # ] = self.lidar.max_distance
        obs_upper_bound[:, -ARVADrone.N_ACTIONS * self.ACTION_BUFFER_SIZE :] = 1

        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    def _computeObs(self):
        self.intensities = (
            self.arva_transmitter.signal_intensity(self.pos).reshape(1, -1).T
        )
        # _, lidar_distances = self.lidar.hit_distances(self.pos)
        action_buffer = np.concatenate(self.action_buffer).reshape(
            self.ACTION_BUFFER_SIZE, self.NUM_DRONES, self.N_ACTIONS
        )
        action_buffer = action_buffer.transpose(1, 0, 2).reshape(
            self.NUM_DRONES, self.ACTION_BUFFER_SIZE * self.N_ACTIONS
        )
        observation = np.hstack(
            (
                self.pos,
                self.rpy,
                self.vel,
                self.ang_v,
                self.intensities,
                # lidar_distances,
                action_buffer,
            )
        )
        return observation

    def _computeReward(self):
        return self.intensities[0]

    def _computeTerminated(self):
        dist_to_arva = np.linalg.norm(self.pos[0] - self.arva_transmitter.position)
        return dist_to_arva < TARGET_DISTANCE_TO_ARVA

    def _computeTruncated(self):
        # When tilted or off bounds
        return self.__is_off_bounds() or self.__is_tilted()

    def _computeInfo(self):
        hits, distances = self.lidar.hit_distances(self.pos)

        return {"rotation": self.rpy[0, :]}

    def __is_tilted(self):
        return (
            abs(self.rpy[0][ROLL]) > MAX_ALLOWED_TILT
            or abs(self.rpy[0][PITCH]) > MAX_ALLOWED_TILT
        )

    def __is_off_bounds(self):
        return (
            abs(self.pos[0][X]) > MAX_DISTANCE
            or abs(self.pos[0][Y]) > MAX_DISTANCE
            or abs(self.pos[0][Z]) > MAX_ALLOWED_HEIGHT
        )


if __name__ == "__main__":
    environment = ARVADrone(
        num_drones=1,
        gui=True,
        obstacles=False,
        user_debug_gui=False,
        lidar_sensors=4,
        lidar_range=3,
    )

    # Action represents the RPMs in the 4 rotors
    action = np.asarray([[0, 0, 0, 0]])

    while True:

        obs, reward, term, trunc, info = environment.step(action)
        print(info)
