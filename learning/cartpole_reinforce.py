import gymnasium as gym
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
import numpy as np
import argparse

GAMMA = 0.99
LEARNING_RATE = 0.01
BATCH_SIZE = 4
DEVICE = torch.device("mps")


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


class VPG(nn.Module):
    def __init__(self, input_size, output_size):
        super(VPG, self).__init__()
        self.net = nn.Sequential(
            XavierLinear(input_size, 128),
            nn.ReLU(),
            XavierLinear(128, output_size),
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=0)


def run_episode(model, env, render=False):
    obs = env.reset()[0]
    if render:
        env.render()
    obs = torch.Tensor(env.reset()[0]).to(DEVICE)
    te = tr = False
    rewards, outputs, actions = [], [], []
    while not (te or tr):
        probs = model(obs)
        action = probs.multinomial(1).item()
        obs, r, te, tr, _ = env.step(action)
        obs = torch.Tensor(obs).to(DEVICE)
        r = -1 if te else 0
        rewards.append(r)
        outputs.append(probs)
        actions.append(action)
    return (
        torch.Tensor(rewards).to(DEVICE),
        torch.concatenate(outputs).reshape(len(rewards), 2),
        actions,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--episodes")
    parser.add_argument("-f", "--final", action="store_true")
    args = parser.parse_args()
    env = gym.make("CartPole-v1", render_mode="human")
    model = VPG(env.observation_space.shape[0], 2).to(DEVICE)
    if args.final:
        state_dict = torch.load(f"results/final_model_388_eps.pt")
    else:
        state_dict = torch.load(f"results/model_{args.episodes}_eps.pt")

    model.load_state_dict(state_dict)
    _ = run_episode(model, env, render=True)
