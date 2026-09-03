"""On-policy rollout storage for Martha's PPO implementation."""

import numpy as np
import torch

from .network import RECURRENT_HIDDEN_SIZE, RecurrentState


class RolloutBuffer:
    """Store transitions until the next PPO update."""

    def __init__(self):
        """
        Initialize an empty on-policy rollout.

        The buffer is cleared after every PPO update.
        """
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.next_values = []
        self.terminateds = []
        self.episode_ends = []
        self.policy_masks = []
        self.episode_starts = []
        self.actor_h = []
        self.actor_c = []
        self.critic_h = []
        self.critic_c = []

    @staticmethod
    def _as_array(x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float32)

    @staticmethod
    def _as_scalar(x):
        return float(RolloutBuffer._as_array(x).reshape(-1)[0])

    @staticmethod
    def _remove_batch_dim(x):
        x = RolloutBuffer._as_array(x)
        if x.ndim > 1 and x.shape[0] == 1:
            return x[0]
        return x

    @staticmethod
    def _recurrent_components(state):
        if state is None:
            return tuple(
                np.zeros(RECURRENT_HIDDEN_SIZE, dtype=np.float32)
                for _ in range(4)
            )
        if not isinstance(state, RecurrentState):
            raise TypeError("recurrent_state must be a RecurrentState")
        components = []
        for value in state:
            array = RolloutBuffer._as_array(value).reshape(-1)
            if array.shape != (RECURRENT_HIDDEN_SIZE,):
                raise ValueError(
                    "single-environment recurrent tensors must contain "
                    f"{RECURRENT_HIDDEN_SIZE} values"
                )
            components.append(array.copy())
        return tuple(components)

    def store(
        self,
        state,
        action,
        logprob,
        reward,
        value,
        *,
        next_value,
        terminated,
        episode_end,
        policy_sample=True,
        recurrent_state=None,
        episode_start=None,
    ):
        """Store one truncation-safe transition."""
        if episode_start is None:
            episode_start = not self.states or bool(self.episode_ends[-1])
        recurrent = self._recurrent_components(recurrent_state)
        self.states.append(self._remove_batch_dim(state))
        self.actions.append(self._remove_batch_dim(action))
        self.logprobs.append(self._as_scalar(logprob))
        self.rewards.append(float(reward))
        self.values.append(self._as_scalar(value))
        self.next_values.append(self._as_scalar(next_value))
        self.terminateds.append(float(terminated))
        self.episode_ends.append(float(episode_end))
        self.policy_masks.append(float(bool(policy_sample)))
        self.episode_starts.append(float(bool(episode_start)))
        self.actor_h.append(recurrent[0])
        self.actor_c.append(recurrent[1])
        self.critic_h.append(recurrent[2])
        self.critic_c.append(recurrent[3])

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.values.clear()
        self.next_values.clear()
        self.terminateds.clear()
        self.episode_ends.clear()
        self.policy_masks.clear()
        self.episode_starts.clear()
        self.actor_h.clear()
        self.actor_c.clear()
        self.critic_h.clear()
        self.critic_c.clear()

    def get_training_batch(self):
        """Return all rollout tensors needed by truncation-safe GAE."""
        states = torch.as_tensor(np.asarray(self.states, dtype=np.float32))
        actions = torch.as_tensor(np.asarray(self.actions, dtype=np.float32))
        if actions.ndim == 1:
            actions = actions.unsqueeze(-1)

        return (
            states,
            actions,
            torch.as_tensor(self.logprobs, dtype=torch.float32).view(-1, 1),
            torch.as_tensor(self.rewards, dtype=torch.float32).view(-1, 1),
            torch.as_tensor(self.values, dtype=torch.float32).view(-1, 1),
            torch.as_tensor(self.next_values, dtype=torch.float32).view(-1, 1),
            torch.as_tensor(self.terminateds, dtype=torch.float32).view(-1, 1),
            torch.as_tensor(self.episode_ends, dtype=torch.float32).view(-1, 1),
            torch.as_tensor(self.policy_masks, dtype=torch.float32).view(-1, 1),
        )

    def get_recurrent_training_batch(self):
        """Return rollout values plus ordered LSTM state and reset markers."""
        return self.get_training_batch() + (
            torch.as_tensor(
                self.episode_starts,
                dtype=torch.float32,
            ).view(-1, 1),
            torch.as_tensor(np.asarray(self.actor_h, dtype=np.float32)),
            torch.as_tensor(np.asarray(self.actor_c, dtype=np.float32)),
            torch.as_tensor(np.asarray(self.critic_h, dtype=np.float32)),
            torch.as_tensor(np.asarray(self.critic_c, dtype=np.float32)),
        )

    def __len__(self):
        return len(self.states)
