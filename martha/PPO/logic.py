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
        self.actor_parameters = list(self.network.actor_parameters())
        self.critic_parameters = list(self.network.critic_parameters())

    @staticmethod
    def empty_stats():
        """Return the complete finite metric schema before the first update."""
        return {
            "loss": 0.0,
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "explained_variance": 0.0,
            "policy_std": 0.0,
            "actor_saturation": 0.0,
            "critic_saturation": 0.0,
            "updates": 0,
        }

    def _clip_gradients(self):
        """Clip actor and critic independently so neither suppresses the other."""
        nn.utils.clip_grad_norm_(
            self.actor_parameters,
            self.max_grad_norm,
        )
        nn.utils.clip_grad_norm_(
            self.critic_parameters,
            self.max_grad_norm,
        )

    def compute_gae(
        self,
        rewards,
        values,
        next_values,
        terminateds,
        episode_ends,
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
        next_values = next_values.view(-1, 1)
        terminateds = terminateds.view(-1, 1)
        episode_ends = episode_ends.view(-1, 1)

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
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_fraction = (
            (torch.abs(ratio - 1.0) > self.eps).float().mean()
        )

        return (
            loss,
            actor_loss,
            critic_loss,
            entropy_loss,
            approx_kl,
            clip_fraction,
        )

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
        stats = self.empty_stats()

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(batch_size, device=states.device)

            for start in range(0, batch_size, minibatch_size):
                batch_indices = indices[start:start + minibatch_size]
                loss_values = self._loss_for_batch(
                    states[batch_indices],
                    actions[batch_indices],
                    old_logprobs[batch_indices],
                    advantages[batch_indices],
                    returns[batch_indices],
                )
                (
                    loss,
                    actor_loss,
                    critic_loss,
                    entropy_loss,
                    approx_kl,
                    clip_fraction,
                ) = loss_values

                self.optimizer.zero_grad()
                loss.backward()
                self._clip_gradients()
                self.optimizer.step()

                stats["loss"] += loss.item()
                stats["actor_loss"] += actor_loss.item()
                stats["critic_loss"] += critic_loss.item()
                stats["entropy"] += entropy_loss.item()
                stats["approx_kl"] += approx_kl.item()
                stats["clip_fraction"] += clip_fraction.item()
                stats["updates"] += 1

        update_count = max(int(stats["updates"]), 1)
        for name in (
            "loss",
            "actor_loss",
            "critic_loss",
            "entropy",
            "approx_kl",
            "clip_fraction",
        ):
            stats[name] /= update_count

        with torch.no_grad():
            _, _, updated_values = self.network.evaluate_actions(
                states,
                actions,
            )
            return_variance = torch.var(returns, unbiased=False)
            if return_variance.item() > 1e-8:
                residual_variance = torch.var(
                    returns - updated_values,
                    unbiased=False,
                )
                stats["explained_variance"] = float(
                    (1.0 - residual_variance / return_variance).item()
                )
            else:
                stats["explained_variance"] = 0.0
        stats.update(self.network.diagnostic_stats(states))
        return stats

    def train_buffer(self, buffer):
        """Train on the complete rollout and clear the on-policy buffer."""
        return self.train_buffers([buffer])

    def train_buffers(self, buffers):
        """Train on independent trajectories without leaking GAE across envs."""
        active_buffers = [buffer for buffer in buffers if len(buffer) > 0]
        if not active_buffers:
            return self.empty_stats()

        batches = []
        for buffer in active_buffers:
            batch = buffer.get_training_batch()
            advantages, returns = self.compute_gae(
                rewards=batch[3],
                values=batch[4],
                next_values=batch[5],
                terminateds=batch[6],
                episode_ends=batch[7],
            )
            batches.append((*batch[:3], advantages, returns))
        states, actions, old_logprobs, advantages, returns = (
            torch.cat([batch[index] for batch in batches], dim=0)
            for index in range(5)
        )
        stats = self.update(states, actions, old_logprobs, advantages, returns)
        for buffer in active_buffers:
            buffer.clear()
        return stats
