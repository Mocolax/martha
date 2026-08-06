"""PPO-Clip optimization and generalized advantage estimation."""

import torch
import torch.nn as nn
import torch.optim as optim


class PPOLogic:
    """Train an actor-critic network with minibatched PPO-Clip updates."""

    def __init__(
        self,
        network,
        lr=3e-4,
        eps=0.2,
        gamma=0.99,
        lam=0.95,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        ppo_epochs=8,
        minibatch_size=256,
    ):
        """Initialize PPO-Clip hyperparameters and the Adam optimizer."""
        self.network = network
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.eps = eps
        self.gamma = gamma
        self.lam = lam
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size

    def compute_gae(
        self,
        rewards,
        values,
        dones=None,
        last_value=None,
        *,
        next_values=None,
        terminateds=None,
        episode_ends=None,
    ):
        """
        Calcula ventajas con GAE y retornos para entrenar el critico.

        ``terminateds`` controls value bootstrap, while ``episode_ends`` cuts
        the recursive GAE trace.  This distinction prevents time limits from
        being treated as absorbing states without leaking advantages into the
        next reset episode.
        """
        rewards = rewards.view(-1, 1)
        values = values.view(-1, 1)
        if episode_ends is None:
            if dones is None:
                raise ValueError("episode_ends or dones is required")
            episode_ends = dones
        if terminateds is None:
            terminateds = dones if dones is not None else episode_ends
        terminateds = terminateds.view(-1, 1)
        episode_ends = episode_ends.view(-1, 1)

        if next_values is None:
            if last_value is None:
                last_value = torch.zeros_like(values[0])
            else:
                last_value = torch.as_tensor(
                    last_value,
                    dtype=values.dtype,
                    device=values.device,
                ).view(1)
            next_values = torch.cat((values[1:], last_value.view(1, 1)), dim=0)
        else:
            next_values = next_values.view(-1, 1)

        advantages = torch.zeros_like(rewards)
        last_gae_lam = torch.zeros_like(values[0])
        for t in reversed(range(len(rewards))):
            bootstrap_mask = 1.0 - terminateds[t]
            trace_mask = 1.0 - episode_ends[t]
            delta = (
                rewards[t]
                + self.gamma * next_values[t] * bootstrap_mask
                - values[t]
            )
            last_gae_lam = (
                delta
                + self.gamma * self.lam * trace_mask * last_gae_lam
            )
            advantages[t] = last_gae_lam

        returns = advantages + values
        return advantages.detach(), returns.detach()

    def _prepare_batch(self, states, actions, old_logprobs, advantages, returns):
        device = next(self.network.parameters()).device
        states = states.to(device)
        actions = actions.to(device)
        old_logprobs = old_logprobs.to(device).view(-1, 1)
        advantages = advantages.to(device).view(-1, 1)
        returns = returns.to(device).view(-1, 1)

        advantage_std = advantages.std(unbiased=False)
        if torch.isfinite(advantage_std).item() and advantage_std.item() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantage_std + 1e-8)
        else:
            advantages = advantages - advantages.mean()

        return states, actions, old_logprobs, advantages, returns

    def _loss_for_batch(self, states, actions, old_logprobs, advantages, returns):
        new_logprobs, entropy, values = self.network.evaluate_actions(states, actions)
        log_ratio = torch.clamp(new_logprobs - old_logprobs, min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.eps, 1.0 + self.eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        critic_loss = nn.MSELoss()(values, returns)
        entropy_loss = entropy.mean()
        loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy_loss

        return loss, actor_loss, critic_loss, entropy_loss

    def train_step(self, states, actions, old_logprobs, advantages, returns):
        """Ejecuta un paso de optimizacion PPO sobre un batch completo."""
        states, actions, old_logprobs, advantages, returns = self._prepare_batch(
            states,
            actions,
            old_logprobs,
            advantages,
            returns,
        )
        loss, actor_loss, critic_loss, entropy_loss = self._loss_for_batch(
            states,
            actions,
            old_logprobs,
            advantages,
            returns,
        )

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return loss.item(), actor_loss.item(), critic_loss.item(), entropy_loss.item()

    def update(self, states, actions, old_logprobs, advantages, returns):
        """Actualiza PPO con multiples epocas y minibatches."""
        states, actions, old_logprobs, advantages, returns = self._prepare_batch(
            states,
            actions,
            old_logprobs,
            advantages,
            returns,
        )

        batch_size = states.shape[0]
        minibatch_size = min(self.minibatch_size, batch_size)
        stats = {
            "loss": 0.0,
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
            "updates": 0,
        }

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(batch_size, device=states.device)

            for start in range(0, batch_size, minibatch_size):
                batch_indices = indices[start:start + minibatch_size]
                loss, actor_loss, critic_loss, entropy_loss = self._loss_for_batch(
                    states[batch_indices],
                    actions[batch_indices],
                    old_logprobs[batch_indices],
                    advantages[batch_indices],
                    returns[batch_indices],
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                stats["loss"] = loss.item()
                stats["actor_loss"] = actor_loss.item()
                stats["critic_loss"] = critic_loss.item()
                stats["entropy"] = entropy_loss.item()
                stats["updates"] += 1

        return stats

    def train_buffer(self, buffer, last_value=None):
        """Train on the complete rollout and clear the on-policy buffer."""
        if len(buffer) == 0:
            return {
                "loss": 0.0,
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "entropy": 0.0,
                "updates": 0,
            }

        (
            states,
            actions,
            old_logprobs,
            rewards,
            values,
            next_values,
            terminateds,
            episode_ends,
        ) = buffer.get_training_batch()
        advantages, returns = self.compute_gae(
            rewards=rewards,
            values=values,
            next_values=next_values,
            terminateds=terminateds,
            episode_ends=episode_ends,
        )
        stats = self.update(states, actions, old_logprobs, advantages, returns)
        buffer.clear()
        return stats
