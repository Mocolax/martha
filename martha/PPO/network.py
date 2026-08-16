"""Paper-inspired multi-branch actor-critic network for Martha PPO."""

import math

import torch
import torch.nn as nn
from torch.distributions import Normal

from .actions import ACTION_SIZE
from .observations import (
    LASER_SECTORS,
    OBSERVATION_FRAME_SIZE,
    OBSERVATION_HISTORY_FRAMES,
    OBSERVATION_SIZE,
)


FUSED_FEATURE_SIZE = 384


class NavigationFeatureExtractor(nn.Module):
    """Encode temporal LiDAR, goal and velocity branches independently."""

    def __init__(self) -> None:
        super().__init__()
        self.laser_branch = nn.Sequential(
            nn.Conv1d(
                OBSERVATION_HISTORY_FRAMES,
                16,
                kernel_size=6,
                stride=3,
            ),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 4, 256),
            nn.ReLU(),
        )
        self.orientation_branch = nn.Sequential(
            nn.Linear(OBSERVATION_HISTORY_FRAMES * 2, 32),
            nn.ReLU(),
        )
        self.distance_branch = nn.Sequential(
            nn.Linear(OBSERVATION_HISTORY_FRAMES, 16),
            nn.ReLU(),
        )
        self.velocity_branch = nn.Sequential(
            nn.Linear(OBSERVATION_HISTORY_FRAMES * 3, 32),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(256 + 32 + 16 + 32, FUSED_FEATURE_SIZE),
            nn.ReLU(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Split the canonical frame-major observation and fuse its branches."""
        if state.ndim != 2 or state.shape[1] != OBSERVATION_SIZE:
            raise ValueError(
                f"state must have shape [N, {OBSERVATION_SIZE}]"
            )
        frames = state.reshape(
            state.shape[0],
            OBSERVATION_HISTORY_FRAMES,
            OBSERVATION_FRAME_SIZE,
        )
        laser = frames[:, :, :LASER_SECTORS]
        distance = frames[:, :, LASER_SECTORS]
        orientation = frames[:, :, LASER_SECTORS + 1:LASER_SECTORS + 3]
        velocity = frames[:, :, LASER_SECTORS + 3:]

        features = torch.cat(
            (
                self.laser_branch(laser),
                self.orientation_branch(orientation.flatten(start_dim=1)),
                self.distance_branch(distance),
                self.velocity_branch(velocity.flatten(start_dim=1)),
            ),
            dim=1,
        )
        return self.fusion(features)


class ActorCritic(nn.Module):
    """Independent Gaussian actor and scalar critic navigation networks."""

    def __init__(self) -> None:
        super().__init__()
        self.tanh_epsilon = 1e-6
        self.actor_feature_extractor = NavigationFeatureExtractor()
        self.actor_mean = nn.Linear(FUSED_FEATURE_SIZE, ACTION_SIZE)
        self.actor_logstd = nn.Parameter(
            torch.full((1, ACTION_SIZE), math.log(0.5))
        )
        self.critic_feature_extractor = NavigationFeatureExtractor()
        self.critic_value = nn.Linear(FUSED_FEATURE_SIZE, 1)

    def actor_parameters(self):
        """Yield only policy parameters for independent gradient clipping."""
        yield from self.actor_feature_extractor.parameters()
        yield from self.actor_mean.parameters()
        yield self.actor_logstd

    def critic_parameters(self):
        """Yield only value-function parameters."""
        yield from self.critic_feature_extractor.parameters()
        yield from self.critic_value.parameters()

    def _distribution(self, state):
        actor_features = self.actor_feature_extractor(state)
        mean = self.actor_mean(actor_features)
        logstd = torch.clamp(self.actor_logstd, min=-5.0, max=2.0)
        std = torch.exp(logstd).expand_as(mean)
        return Normal(mean, std)

    def _value(self, state):
        critic_features = self.critic_feature_extractor(state)
        return self.critic_value(critic_features)

    def forward(self, state):
        """Return the independent policy distribution and value estimate."""
        return self._distribution(state), self._value(state)

    def _squashed_log_prob(self, dist, action):
        """Return the Jacobian-corrected probability of a squashed action."""
        clamped_action = torch.clamp(
            action,
            -1.0 + self.tanh_epsilon,
            1.0 - self.tanh_epsilon,
        )
        raw_action = torch.atanh(clamped_action)
        gaussian_log_prob = dist.log_prob(raw_action).sum(dim=-1, keepdim=True)
        tanh_correction = torch.log(
            1.0 - clamped_action.pow(2) + self.tanh_epsilon
        ).sum(dim=-1, keepdim=True)
        return gaussian_log_prob - tanh_correction

    def evaluate_actions(self, states, actions):
        """Recompute log-probabilities, entropy and values for PPO."""
        dist, values = self.forward(states)
        log_probs = self._squashed_log_prob(dist, actions)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_probs, entropy, values

    def get_value(self, state):
        """Calculate the value used for bootstrap or evaluation."""
        device = next(self.parameters()).device
        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            _, value = self.forward(state)
        return value.cpu()

    def get_actions(self, states, deterministic=False):
        """Return bounded actions, log-probabilities and values for a batch."""
        device = next(self.parameters()).device
        states = torch.as_tensor(states, dtype=torch.float32, device=device)
        if states.ndim == 1:
            states = states.unsqueeze(0)
        if states.ndim != 2 or states.shape[1] != OBSERVATION_SIZE:
            raise ValueError(
                f"states must have shape [N, {OBSERVATION_SIZE}]"
            )

        with torch.no_grad():
            dist, values = self.forward(states)
            raw_actions = dist.mean if deterministic else dist.sample()
            actions = torch.tanh(raw_actions)
            log_probs = self._squashed_log_prob(dist, actions)
        return actions.cpu(), log_probs.cpu(), values.cpu()

    def get_action(self, state, deterministic=False):
        """Return one continuous action bounded to ``[-1, 1]``."""
        action, log_prob, value = self.get_actions(
            state,
            deterministic=deterministic,
        )
        return action.squeeze(0), log_prob, value

    def diagnostic_stats(self, states):
        """Measure ReLU inactivity and the current exploration scale."""
        device = next(self.parameters()).device
        states = torch.as_tensor(states, dtype=torch.float32, device=device)
        if states.ndim == 1:
            states = states.unsqueeze(0)
        with torch.no_grad():
            actor_features = self.actor_feature_extractor(states)
            critic_features = self.critic_feature_extractor(states)
            policy_std = torch.exp(
                torch.clamp(self.actor_logstd, min=-5.0, max=2.0)
            ).mean()
        return {
            "policy_std": float(policy_std.item()),
            "actor_inactive_relu": float(
                (actor_features == 0.0).float().mean().item()
            ),
            "critic_inactive_relu": float(
                (critic_features == 0.0).float().mean().item()
            ),
        }
