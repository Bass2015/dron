import gymnasium as gym
import torch
import ale_py
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
import numpy as np
import plotly.express as px


GAMMA = 0.99
LEARNING_RATE = 0.01
BATCH_SIZE = 64
ENTROPY_BETA = 0.01
NUM_ENVS = 50
REWARD_STEPS = 6  # How many steps ahead we take to approximate the total discounted reward for every action.
CLIP_GRAD = 0.1  # Prevents gradients from becoming too large. If L2 norm of gradients is bigger than this number, grad vector is clipped to this value.
ADAM_EPS = 1e-3

DEVICE = torch.device("mps")


def preprocess(obs):
    obs = obs[35:195]  # crop
    obs = obs[:, :, 0]  # downsample by factor of 2
    obs[obs == 144] = 0.0  # erase background (background type 1)
    obs[obs == 109] = 0.0  # erase background (background type 2)
    obs[obs != 0] = 1.0  # everything else (paddles, ball) just set to 1
    obs = obs.astype(np.float32)
    obs = torch.Tensor(obs).to(DEVICE)
    return obs


class XavierLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        xavier_uniform_(self.weight)


class XavierConv2d(nn.Conv2d):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size, stride
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride)
        xavier_uniform_(self.weight)


class A2C(nn.Module):
    def __init__(self, n_actions):
        super(A2C, self).__init__()
        self.conv = nn.Sequential(
            XavierConv2d(3, 32, kernel_size=8, stride=4),  # in 160x160, out 39x39
            nn.ReLU(),
            XavierConv2d(32, 64, kernel_size=5, stride=2),  # in 39x39, out 18x18
            nn.ReLU(),
            XavierConv2d(64, 64, kernel_size=2, stride=1),  # in 8x8, out 17x17
            nn.ReLU(),
        )
        self.flatter = nn.Flatten()
        self.policy = nn.Sequential(
            XavierLinear(18496, 512), nn.ReLU(), XavierLinear(512, n_actions)
        )
        self.value = nn.Sequential(
            XavierLinear(18496, 512), nn.ReLU(), XavierLinear(512, 1)
        )

    def forward(self, x):
        conv_out = self.conv(x)
        flat_conv_out = conv_out.view(x.size()[0], -1)
        probs = F.softmax(self.policy(flat_conv_out), dim=1)
        value_estimate = self.value(flat_conv_out)
        return probs, value_estimate


def run_episode(model, env, render=False):
    raw_state = preprocess(env.reset()[0])
    if render:
        env.render()
    acc_state = torch.zeros_like(raw_state).tile(
        3, 1, 1
    )  # Injects temporal dimension to the state
    acc_state[-1] = raw_state
    te = tr = False
    rewards, outputs, actions, value_estimates = [], [], [], []
    while not (te or tr):
        probs, value_estimate = model(acc_state[None, :, :, :])
        action = probs[0].multinomial(1).item()
        raw_state, r, te, tr, _ = env.step(action)
        raw_state = preprocess(raw_state)
        acc_state = acc_state.roll(-1, dims=0)
        acc_state[-1] = raw_state
        rewards.append(r)
        outputs.append(probs)
        actions.append(action)
        value_estimates.append(value_estimate)
    return (
        torch.Tensor(rewards).to(DEVICE),
        torch.concatenate(outputs).reshape(len(rewards), 6),
        actions,
        torch.concatenate(value_estimates).reshape(-1),
    )


if __name__ == "__main__":
    env = gym.make("PongNoFrameskip-v4", render_mode="human")
    model = A2C(n_actions=env.action_space.n).to(DEVICE)
    state_dict = torch.load("learning/Episode_56_data/model_state_dict.pt")
    model.load_state_dict(state_dict)
    run_episode(model, env, render=True)
