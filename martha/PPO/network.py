"""Actor-critic neural network used by Martha's continuous PPO policy."""

import math
import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """Independent Gaussian actor and scalar critic networks."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.tanh_epsilon = 1e-6

        self.actor_feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.full((1, action_dim), math.log(0.5)))

        self.critic_feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.critic_value = nn.Linear(hidden_dim, 1)

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
        """
        Log-prob de una accion ya acotada con tanh.

        PPO reevalua la accion enviada al entorno, asi que se invierte tanh
        con atanh y se corrige por el jacobiano de la transformacion.
        """
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
        """Recalcula log-prob, entropia y valor para acciones de rollout."""
        dist, values = self.forward(states)
        log_probs = self._squashed_log_prob(dist, actions)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_probs, entropy, values

    def get_value(self, state):
        """Calcula V(s) para bootstrap o evaluacion."""
        device = next(self.parameters()).device
        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        if state.ndim == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            _, value = self.forward(state)
        return value.cpu()

    def get_actions(self, states, deterministic=False):
        """Return actions, log-probabilities and values for a state batch."""
        device = next(self.parameters()).device
        states = torch.as_tensor(states, dtype=torch.float32, device=device)
        if states.ndim == 1:
            states = states.unsqueeze(0)
        if states.ndim != 2:
            raise ValueError("states must have shape [N, state_dim]")

        with torch.no_grad():
            dist, values = self.forward(states)
            raw_actions = dist.mean if deterministic else dist.sample()
            actions = torch.tanh(raw_actions)
            log_probs = self._squashed_log_prob(dist, actions)
        return actions.cpu(), log_probs.cpu(), values.cpu()

    def get_action(self, state, deterministic=False):
        """Muestrea una accion continua acotada en [-1, 1]."""
        action, log_prob, value = self.get_actions(
            state,
            deterministic=deterministic,
        )
        return action.squeeze(0), log_prob, value

    def diagnostic_stats(self, states):
        """Measure representation saturation and current exploration scale."""
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
            "actor_saturation": float(
                (actor_features.abs() > 0.99).float().mean().item()
            ),
            "critic_saturation": float(
                (critic_features.abs() > 0.99).float().mean().item()
            ),
        }
