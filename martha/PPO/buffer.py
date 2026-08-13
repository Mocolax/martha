"""On-policy rollout storage for Martha's PPO implementation."""

import numpy as np
import torch


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
    ):
        """Store one truncation-safe transition."""
        self.states.append(self._remove_batch_dim(state))
        self.actions.append(self._remove_batch_dim(action))
        self.logprobs.append(self._as_scalar(logprob))
        self.rewards.append(float(reward))
        self.values.append(self._as_scalar(value))
        self.next_values.append(self._as_scalar(next_value))
        self.terminateds.append(float(terminated))
        self.episode_ends.append(float(episode_end))

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.values.clear()
        self.next_values.clear()
        self.terminateds.clear()
        self.episode_ends.clear()

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
        )

    def __len__(self):
        return len(self.states)
