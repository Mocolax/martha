"""Paper-inspired multi-branch actor-critic network for Martha PPO."""

import math
from typing import NamedTuple

import torch
import torch.nn as nn
from torch.distributions import Normal

from .actions import ACTION_SIZE
from .observations import LASER_SECTORS, OBSERVATION_SIZE


FUSED_FEATURE_SIZE = 384
RECURRENT_HIDDEN_SIZE = 128
RECURRENT_NUM_LAYERS = 1

POLICY_ARCHITECTURE = "multibranch_lstm"


class RecurrentState(NamedTuple):
    """Independent actor and critic LSTM state."""

    actor_h: torch.Tensor
    actor_c: torch.Tensor
    critic_h: torch.Tensor
    critic_c: torch.Tensor


class NavigationFeatureExtractor(nn.Module):
    """Encode one navigation frame by modality."""

    def __init__(self) -> None:
        super().__init__()
        # The LiDAR sectors close the circle, so both convolutions wrap around
        # the seam behind the robot instead of treating it as an edge.
        self.laser_convolutions = nn.Sequential(
            nn.Conv1d(
                1,
                16,
                kernel_size=6,
                stride=3,
                padding=3,
                padding_mode="circular",
            ),
            nn.ReLU(),
            nn.Conv1d(
                16,
                32,
                kernel_size=5,
                stride=2,
                padding=2,
                padding_mode="circular",
            ),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            laser_features = self.laser_convolutions(
                torch.zeros(1, 1, LASER_SECTORS)
            ).shape[1]
        self.laser_branch = nn.Sequential(
            self.laser_convolutions,
            nn.Linear(laser_features, 256),
            nn.ReLU(),
        )
        self.orientation_branch = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
        )
        self.distance_branch = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
        )
        self.velocity_branch = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(256 + 32 + 16 + 32, FUSED_FEATURE_SIZE),
            nn.ReLU(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Split the canonical observation frame and fuse its branches."""
        if state.ndim != 2 or state.shape[1] != OBSERVATION_SIZE:
            raise ValueError(
                f"state must have shape [N, {OBSERVATION_SIZE}]"
            )
        laser = state[:, None, :LASER_SECTORS]
        distance = state[:, LASER_SECTORS:LASER_SECTORS + 1]
        orientation = state[:, LASER_SECTORS + 1:LASER_SECTORS + 3]
        velocity = state[:, LASER_SECTORS + 3:]

        features = torch.cat(
            (
                self.laser_branch(laser),
                self.orientation_branch(orientation),
                self.distance_branch(distance),
                self.velocity_branch(velocity),
            ),
            dim=1,
        )
        return self.fusion(features)


class ActorCritic(nn.Module):
    """Independent recurrent Gaussian actor and scalar critic networks."""

    def __init__(self) -> None:
        super().__init__()
        self.tanh_epsilon = 1e-6
        self.actor_feature_extractor = NavigationFeatureExtractor()
        self.actor_lstm = nn.LSTM(
            FUSED_FEATURE_SIZE,
            RECURRENT_HIDDEN_SIZE,
            num_layers=RECURRENT_NUM_LAYERS,
            batch_first=True,
        )
        self.actor_mean = nn.Linear(RECURRENT_HIDDEN_SIZE, ACTION_SIZE)
        self.actor_logstd = nn.Parameter(
            torch.full((1, ACTION_SIZE), math.log(0.4))
        )
        self.critic_feature_extractor = NavigationFeatureExtractor()
        self.critic_lstm = nn.LSTM(
            FUSED_FEATURE_SIZE,
            RECURRENT_HIDDEN_SIZE,
            num_layers=RECURRENT_NUM_LAYERS,
            batch_first=True,
        )
        self.critic_value = nn.Linear(RECURRENT_HIDDEN_SIZE, 1)
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """
        Apply the standard PPO orthogonal initialization.

        Hidden ReLU layers use a sqrt(2) gain, the value head a unit gain and
        the policy mean a 0.01 gain, so the initial policy starts small and
        centred instead of emitting large biased actions.
        """
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LSTM):
                for name, parameter in module.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(parameter)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(parameter)
                    elif "bias" in name:
                        nn.init.constant_(parameter, 0.0)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)
        nn.init.orthogonal_(self.critic_value.weight, gain=1.0)
        nn.init.constant_(self.critic_value.bias, 0.0)

    def actor_parameters(self):
        """Yield only policy parameters for independent gradient clipping."""
        yield from self.actor_feature_extractor.parameters()
        yield from self.actor_lstm.parameters()
        yield from self.actor_mean.parameters()
        yield self.actor_logstd

    def critic_parameters(self):
        """Yield only value-function parameters."""
        yield from self.critic_feature_extractor.parameters()
        yield from self.critic_lstm.parameters()
        yield from self.critic_value.parameters()

    def clamp_policy_std(self, maximum: float) -> None:
        """Apply an in-place ceiling to the learned global action STD."""
        maximum = float(maximum)
        if not math.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("maximum policy STD must be finite and positive")
        with torch.no_grad():
            self.actor_logstd.clamp_(max=math.log(maximum))

    def initial_recurrent_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
    ) -> RecurrentState:
        """Return zeroed actor/critic memory for a new episode."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if device is None:
            device = next(self.parameters()).device
        shape = (RECURRENT_NUM_LAYERS, int(batch_size), RECURRENT_HIDDEN_SIZE)
        zeros = [torch.zeros(shape, dtype=torch.float32, device=device) for _ in range(4)]
        return RecurrentState(*zeros)

    @staticmethod
    def stack_recurrent_states(states: list[RecurrentState]) -> RecurrentState:
        """Join single-environment memories into one environment batch."""
        if not states:
            raise ValueError("at least one recurrent state is required")
        return RecurrentState(
            *(torch.cat(values, dim=1) for values in zip(*states))
        )

    @staticmethod
    def recurrent_state_at(
        state: RecurrentState,
        index: int,
    ) -> RecurrentState:
        """Select one environment while retaining the LSTM batch axis."""
        return RecurrentState(
            *(value[:, index:index + 1].detach() for value in state)
        )

    def _validate_recurrent_state(
        self,
        state: RecurrentState,
        batch_size: int,
    ) -> None:
        expected = (
            RECURRENT_NUM_LAYERS,
            batch_size,
            RECURRENT_HIDDEN_SIZE,
        )
        if not isinstance(state, RecurrentState):
            raise TypeError("recurrent_state must be a RecurrentState")
        if any(tuple(value.shape) != expected for value in state):
            raise ValueError(f"recurrent state tensors must have shape {expected}")

    @staticmethod
    def _reset_memory(
        memory: tuple[torch.Tensor, torch.Tensor],
        episode_start: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keep = (1.0 - episode_start).view(1, -1, 1)
        return memory[0] * keep, memory[1] * keep

    def _run_recurrent_branch(
        self,
        features: torch.Tensor,
        recurrent: nn.LSTM,
        memory: tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        outputs = []
        for step in range(features.shape[1]):
            memory = self._reset_memory(memory, episode_starts[:, step])
            output, memory = recurrent(features[:, step:step + 1], memory)
            outputs.append(output)
        return torch.cat(outputs, dim=1), memory

    def forward_recurrent(
        self,
        states: torch.Tensor,
        recurrent_state: RecurrentState | None = None,
        episode_starts: torch.Tensor | None = None,
    ) -> tuple[Normal, torch.Tensor, RecurrentState]:
        """Evaluate an ordered ``[batch, time, observation]`` sequence."""
        if states.ndim == 2:
            states = states.unsqueeze(1)
        if states.ndim != 3 or states.shape[2] != OBSERVATION_SIZE:
            raise ValueError(
                "states must have shape "
                f"[B, T, {OBSERVATION_SIZE}] or [B, {OBSERVATION_SIZE}]"
            )
        batch_size, sequence_length, _ = states.shape
        if sequence_length <= 0:
            raise ValueError("recurrent sequence cannot be empty")
        if recurrent_state is None:
            recurrent_state = self.initial_recurrent_state(
                batch_size,
                device=states.device,
            )
        self._validate_recurrent_state(recurrent_state, batch_size)
        recurrent_state = RecurrentState(
            *(value.to(device=states.device, dtype=states.dtype) for value in recurrent_state)
        )
        if episode_starts is None:
            episode_starts = torch.zeros(
                (batch_size, sequence_length, 1),
                dtype=states.dtype,
                device=states.device,
            )
        else:
            episode_starts = episode_starts.to(
                device=states.device,
                dtype=states.dtype,
            )
            if episode_starts.ndim == 2:
                episode_starts = episode_starts.unsqueeze(-1)
            if episode_starts.shape != (batch_size, sequence_length, 1):
                raise ValueError(
                    "episode_starts must have shape "
                    f"({batch_size}, {sequence_length}, 1)"
                )

        flattened = states.reshape(batch_size * sequence_length, OBSERVATION_SIZE)
        actor_features = self.actor_feature_extractor(flattened).reshape(
            batch_size,
            sequence_length,
            FUSED_FEATURE_SIZE,
        )
        critic_features = self.critic_feature_extractor(flattened).reshape(
            batch_size,
            sequence_length,
            FUSED_FEATURE_SIZE,
        )
        actor_output, actor_memory = self._run_recurrent_branch(
            actor_features,
            self.actor_lstm,
            (recurrent_state.actor_h, recurrent_state.actor_c),
            episode_starts,
        )
        critic_output, critic_memory = self._run_recurrent_branch(
            critic_features,
            self.critic_lstm,
            (recurrent_state.critic_h, recurrent_state.critic_c),
            episode_starts,
        )
        mean = self.actor_mean(actor_output)
        logstd = torch.clamp(self.actor_logstd, min=-5.0, max=2.0)
        std = torch.exp(logstd).expand_as(mean)
        values = self.critic_value(critic_output)
        next_state = RecurrentState(
            actor_memory[0],
            actor_memory[1],
            critic_memory[0],
            critic_memory[1],
        )
        return Normal(mean, std), values, next_state

    def forward(self, state):
        """Evaluate independent samples with zeroed recurrent memory."""
        dist, values, _ = self.forward_recurrent(state)
        return Normal(dist.mean[:, 0], dist.stddev[:, 0]), values[:, 0]

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
        """Recompute independent-sample PPO values with zeroed memory."""
        dist, values = self.forward(states)
        log_probs = self._squashed_log_prob(dist, actions)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_probs, entropy, values

    def evaluate_action_sequences(
        self,
        states,
        actions,
        recurrent_state,
        episode_starts,
    ):
        """Recompute PPO values without destroying temporal ordering."""
        dist, values, _ = self.forward_recurrent(
            states,
            recurrent_state,
            episode_starts,
        )
        log_probs = self._squashed_log_prob(dist, actions)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_probs, entropy, values

    def get_value(self, state):
        """Calculate a value with zeroed recurrent memory."""
        device = next(self.parameters()).device
        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            _, value = self.forward(state)
        return value.cpu()

    def get_value_recurrent(self, states, recurrent_state):
        """Calculate bootstrap values using the preceding trajectory memory."""
        device = next(self.parameters()).device
        states = torch.as_tensor(states, dtype=torch.float32, device=device)
        if states.ndim == 1:
            states = states.unsqueeze(0)
        with torch.no_grad():
            _, values, _ = self.forward_recurrent(states, recurrent_state)
        return values[:, 0].cpu()

    def get_actions(self, states, deterministic=False):
        """Return independent actions with zeroed recurrent memory."""
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

    def get_actions_recurrent(
        self,
        states,
        recurrent_state,
        *,
        episode_starts=None,
        deterministic=False,
    ):
        """Return actions and updated memory for an environment batch."""
        device = next(self.parameters()).device
        states = torch.as_tensor(states, dtype=torch.float32, device=device)
        if states.ndim == 1:
            states = states.unsqueeze(0)
        if states.ndim != 2 or states.shape[1] != OBSERVATION_SIZE:
            raise ValueError(
                f"states must have shape [N, {OBSERVATION_SIZE}]"
            )
        if episode_starts is not None:
            episode_starts = torch.as_tensor(
                episode_starts,
                dtype=torch.float32,
                device=device,
            ).reshape(states.shape[0], 1, 1)
        with torch.no_grad():
            dist, values, next_state = self.forward_recurrent(
                states,
                recurrent_state,
                episode_starts,
            )
            dist = Normal(dist.mean[:, 0], dist.stddev[:, 0])
            values = values[:, 0]
            raw_actions = dist.mean if deterministic else dist.sample()
            actions = torch.tanh(raw_actions)
            log_probs = self._squashed_log_prob(dist, actions)
        return (
            actions.cpu(),
            log_probs.cpu(),
            values.cpu(),
            RecurrentState(*(value.detach() for value in next_state)),
        )

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
