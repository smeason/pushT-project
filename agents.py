import copy
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D
import numpy as np
import cv2
from pathlib import Path
import torch
import torch.nn as nn
from torch.distributions import Normal
import time
import gymnasium as gym
import gym_pusht
from gymnasium.wrappers import FrameStackObservation
from gymnasium.wrappers import FlattenObservation
from gymnasium.wrappers import GrayscaleObservation
from gymnasium import ActionWrapper, spaces

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, action_std: float = 0.5):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), np.log(action_std)))

    def get_distribution(self, state: torch.Tensor) -> Normal:
        mean = self.actor(state)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def act(self, state: torch.Tensor):
        dist = self.get_distribution(state)
        action = dist.sample()
        return action, dist.log_prob(action).sum(dim=-1), self.critic(state).squeeze(-1)

    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        dist = self.get_distribution(state)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(state).squeeze(-1)
        return log_prob, value, entropy

class ActorCriticWithHistory(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, history_len: int=8, action_std: float = 0.3):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 100),
            nn.Tanh(),
            nn.Linear(100, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 100),
            nn.Tanh(),
            nn.Linear(100, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), np.log(action_std)))

    def get_distribution(self, state: torch.Tensor) -> Normal:
        mean = self.actor(state)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def act(self, state: torch.Tensor):
        dist = self.get_distribution(state)
        action = dist.sample()
        return action, dist.log_prob(action).sum(dim=-1), self.critic(state).squeeze(-1)

    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        dist = self.get_distribution(state)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(state).squeeze(-1)
        return log_prob, value, entropy

class ActorCriticDiscreteWithHistory(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, history_len: int=8, num_bins: int=11):
        super().__init__()
        self.num_bins = num_bins
        self.action_dim = action_dim
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 100),
            nn.Tanh(),
            nn.Linear(100, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim * num_bins),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 100),
            nn.Tanh(),
            nn.Linear(100, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def get_distribution(self, state: torch.Tensor):
        logits = self.actor(state)
        # Split logits into action_dim parts
        logits_split = torch.split(logits, self.num_bins, dim=-1)
        dists = [torch.distributions.Categorical(logits=l) for l in logits_split]
        return dists

    def act(self, state: torch.Tensor):
        dists = self.get_distribution(state)
        actions = [dist.sample() for dist in dists]
        log_probs = [dist.log_prob(a) for dist, a in zip(dists, actions)]

        action = torch.stack(actions, dim=-1)
        log_prob = torch.stack(log_probs, dim=-1).sum(dim=-1)
        return action, log_prob, self.critic(state).squeeze(-1)

    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        dists = self.get_distribution(state)
        log_probs = []
        entropies = []
        for i, dist in enumerate(dists):
            log_probs.append(dist.log_prob(action[..., i]))
            entropies.append(dist.entropy())

        log_prob = torch.stack(log_probs, dim=-1).sum(dim=-1)
        entropy = torch.stack(entropies, dim=-1).sum(dim=-1)
        value = self.critic(state).squeeze(-1)
        return log_prob, value, entropy


# @title
class ActorCriticCNN(nn.Module):
    def __init__(self, in_channels: int = 4, action_dim: int = 2, action_std: float = 0.4):
        super().__init__()

        # Shared feature extractor for spatial processing of 96x96 frames
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute flattened feature dimension dynamically for a 96x96 input image
        # Shape progression: 96x96 -> (with 8x8 s=4) 23x23 -> (with 4x4 s=2) 10x10 -> (with 3x3 s=1) 8x8
        # Flat features: 64 * 8 * 8 = 4096
        flat_features_dim = 4096

        # Policy Head (Actor)
        self.actor = nn.Sequential(
            nn.Linear(flat_features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh() # Continuous actions bound between [-1, 1]
        )

        # Value Head (Critic)
        self.critic_net = nn.Sequential(
            nn.Linear(flat_features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.log_std = nn.Parameter(torch.full((action_dim,), np.log(action_std)))

    def critic(self, state: torch.Tensor) -> torch.Tensor:
        # If input state is a 4D raw image batch [Batch, Channels, Height, Width], extract features first
        if state.ndim == 4:
            features = self.feature_extractor(state)
        else:
            features = state
        return self.critic_net(features)

    def get_distribution(self, state: torch.Tensor) -> Normal:
        # Expects state shape: [Batch, Channels, Height, Width]
        features = self.feature_extractor(state)
        mean = self.actor(features)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def act(self, state: torch.Tensor):
        features = self.feature_extractor(state)
        dist = Normal(self.actor(features), self.log_std.exp().expand_as(self.actor(features)))
        action = dist.sample()
        return action, dist.log_prob(action).sum(dim=-1), self.critic_net(features).squeeze(-1)

    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        features = self.feature_extractor(state)
        mean = self.actor(features)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic_net(features).squeeze(-1)
        return log_prob, value, entropy