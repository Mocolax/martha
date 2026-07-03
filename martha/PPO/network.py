import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        hidden_half = hidden_dim // 2
        self.tanh_epsilon = 1e-6

        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Politica continua: predice la media gaussiana cruda.
        # La accion enviada al entorno se acota despues con tanh.
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_half),
            nn.ReLU(),
            nn.Linear(hidden_half, 1),
        )

    def forward(self, state):
        """Devuelve la distribucion gaussiana cruda y V(s)."""
        features = self.feature_extractor(state)
        mean = self.actor_mean(features)
        logstd = torch.clamp(self.actor_logstd, min=-5.0, max=2.0)
        std = torch.exp(logstd).expand_as(mean)
        dist = Normal(mean, std)
        value = self.critic(features)
        return dist, value

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

    def get_action(self, state, deterministic=False):
        """Muestrea una accion continua acotada en [-1, 1]."""
        device = next(self.parameters()).device
        state = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            dist, value = self.forward(state)
            raw_action = dist.mean if deterministic else dist.sample()
            action = torch.tanh(raw_action)
            log_prob = self._squashed_log_prob(dist, action)

        return action.squeeze(0).cpu(), log_prob.cpu(), value.cpu()
