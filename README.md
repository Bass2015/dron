# dron: a search-and-rescue drone that learns to find an avalanche victim

A reinforcement-learning environment where a quadcopter has to find a person buried under snow by following the signal of their avalanche transceiver (ARVA). Built on [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) and Gymnasium.

**Status: work in progress.** The environment, the transceiver signal model and the reward are in place. Training an agent on the drone is the next step. The `learning/` folder has the policy-gradient methods I wrote from scratch to get there.

## The environment (`dron_env.py`)

- **Scene:** a Crazyflie 2.X drone starts at a random point within ±5 m of the buried transceiver (shown as a duck).
- **Signal model (`arva.py`):** the transceiver is modelled as a magnetic dipole, following Piniés and Tardós ([IEEE, 2006](https://ieeexplore.ieee.org/document/1642314)). The drone measures the field intensity at its position.
- **Observation:** position, orientation, velocity, angular velocity, signal intensity, plus a buffer of the last half second of actions.
- **Action:** the RPM of each of the four rotors.
- **Reward:** the signal intensity at the drone's position, a dense signal the drone can climb towards the victim.
- **Episode ends** when the drone gets within 0.5 m of the transceiver. It is cut short if the drone leaves the area (±5 m, 2 m high) or tilts more than 0.4 rad.
- **Lidar (`lidar.py`):** ray-cast distance sensors for obstacle avoidance. Implemented but not yet part of the observation.

Run the environment with the PyBullet GUI:

```bash
python dron_env.py
```

## Policy-gradient experiments (`learning/`)

Written from scratch in PyTorch on standard Gymnasium tasks before moving to the drone:

- `cartpole_reinforce.py`: REINFORCE (vanilla policy gradient) on CartPole
- `01b_CartPoleA2C.ipynb`, `02_ActorCritic.ipynb`: advantage actor-critic
- `pong_a2c.py`: A2C on Pong
- `results.ipynb`: policy, value and entropy losses and gradient statistics across episodes

`notes.ipynb` has working notes on collision detection and on the dipole field formula.

## Requirements

Python with `gymnasium`, `gym-pybullet-drones`, `pybullet`, `numpy`, `torch` and `plotly` (see `requirements.txt`).

## Related data

The simulator does not use real measurements yet. For field data on signals from transmitters buried under snow, see M. Girolami, F. Mavilia, G. Marrocco, G. M. Bianco, *"An Experimental Dataset for Search and Rescue Operations in Avalanche Scenarios Based on LoRa Technology"*. The dataset is not included in this repository.
