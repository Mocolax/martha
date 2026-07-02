import numpy as np
import torch


class RolloutBuffer:
    def __init__(self):
        """
        Buffer on-policy para guardar las trayectorias recientes del agente.
        Se limpia despues de cada actualizacion de PPO.
        """
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

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

    def store(self, state, action, logprob, reward, done, value):
        """Guarda una transicion de rollout."""
        self.states.append(self._remove_batch_dim(state))
        self.actions.append(self._remove_batch_dim(action))
        self.logprobs.append(self._as_scalar(logprob))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(self._as_scalar(value))

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def get_all(self):
        """
        Devuelve tensores con formas estables:
        states [T, state_dim], actions [T, action_dim],
        logprobs/rewards/dones/values [T, 1].
        """
        states = torch.as_tensor(np.asarray(self.states, dtype=np.float32))
        actions = torch.as_tensor(np.asarray(self.actions, dtype=np.float32))
        if actions.ndim == 1:
            actions = actions.unsqueeze(-1)

        return (
            states,
            actions,
            torch.as_tensor(self.logprobs, dtype=torch.float32).view(-1, 1),
            torch.as_tensor(self.rewards, dtype=torch.float32).view(-1, 1),
            torch.as_tensor(self.dones, dtype=torch.float32).view(-1, 1),
            torch.as_tensor(self.values, dtype=torch.float32).view(-1, 1),
        )

    def __len__(self):
        return len(self.states)
