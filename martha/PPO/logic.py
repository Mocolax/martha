"""PPO-Clip optimization and generalized advantage estimation."""

import torch
import torch.nn as nn
import torch.optim as optim

from .network import RecurrentState


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
        recurrent_sequence_length=32,
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
        self.recurrent_sequence_length = recurrent_sequence_length
        if self.recurrent_sequence_length <= 0:
            raise ValueError("recurrent_sequence_length must be positive")
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
            "actor_inactive_relu": 0.0,
            "critic_inactive_relu": 0.0,
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

    def _prepare_batch(
        self,
        states,
        actions,
        old_logprobs,
        advantages,
        returns,
        policy_masks,
    ):
        device = next(self.network.parameters()).device
        states = states.to(device)
        actions = actions.to(device)
        old_logprobs = old_logprobs.to(device).view(-1, 1)
        advantages = advantages.to(device).view(-1, 1)
        returns = returns.to(device).view(-1, 1)
        policy_masks = policy_masks.to(device).view(-1, 1)

        policy_advantages = advantages[policy_masks > 0.5]
        if policy_advantages.numel() == 0:
            advantages = torch.zeros_like(advantages)
        else:
            advantage_std = policy_advantages.std(unbiased=False)
            if (
                torch.isfinite(advantage_std).item()
                and advantage_std.item() > 1e-8
            ):
                advantages = (
                    advantages - policy_advantages.mean()
                ) / (advantage_std + 1e-8)
            else:
                advantages = advantages - policy_advantages.mean()

        return states, actions, old_logprobs, advantages, returns, policy_masks

    def _loss_for_batch(
        self,
        states,
        actions,
        old_logprobs,
        advantages,
        returns,
        policy_masks,
    ):
        new_logprobs, entropy, values = self.network.evaluate_actions(states, actions)
        log_ratio = torch.clamp(new_logprobs - old_logprobs, min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.eps, 1.0 + self.eps) * advantages
        policy_count = policy_masks.sum().clamp_min(1.0)
        actor_loss = -(
            torch.min(surr1, surr2) * policy_masks
        ).sum() / policy_count

        critic_loss = nn.MSELoss()(values, returns)
        entropy_loss = (entropy * policy_masks).sum() / policy_count
        loss = (
            actor_loss
            + self.value_coef * critic_loss
            - self.entropy_coef * entropy_loss
        )
        approx_kl = (
            ((ratio - 1.0) - log_ratio) * policy_masks
        ).sum() / policy_count
        clip_fraction = (
            (
                (torch.abs(ratio - 1.0) > self.eps).float()
                * policy_masks
            ).sum()
            / policy_count
        )

        return (
            loss,
            actor_loss,
            critic_loss,
            entropy_loss,
            approx_kl,
            clip_fraction,
        )

    def update(
        self,
        states,
        actions,
        old_logprobs,
        advantages,
        returns,
        policy_masks=None,
    ):
        """Actualiza PPO con multiples epocas y minibatches."""
        if policy_masks is None:
            policy_masks = torch.ones_like(advantages)
        if not torch.any(policy_masks > 0.5):
            raise ValueError("PPO update requires at least one policy sample")
        (
            states,
            actions,
            old_logprobs,
            advantages,
            returns,
            policy_masks,
        ) = self._prepare_batch(
            states,
            actions,
            old_logprobs,
            advantages,
            returns,
            policy_masks,
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
                    policy_masks[batch_indices],
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

    @staticmethod
    def _pad_sequence(values, target_length):
        padded = torch.zeros(
            (target_length, *values.shape[1:]),
            dtype=values.dtype,
        )
        padded[:values.shape[0]] = values
        return padded

    def _recurrent_sequences(self, buffers):
        """Build padded ordered chunks without mixing environment histories."""
        chunks = []
        sequence_length = self.recurrent_sequence_length
        for buffer in buffers:
            batch = buffer.get_recurrent_training_batch()
            advantages, returns = self.compute_gae(
                rewards=batch[3],
                values=batch[4],
                next_values=batch[5],
                terminateds=batch[6],
                episode_ends=batch[7],
            )
            ordered = (
                batch[0],
                batch[1],
                batch[2],
                advantages,
                returns,
                batch[8],
                batch[9],
            )
            for start in range(0, len(buffer), sequence_length):
                stop = min(start + sequence_length, len(buffer))
                valid = torch.zeros((sequence_length, 1), dtype=torch.float32)
                valid[:stop - start] = 1.0
                chunks.append(
                    (
                        *(
                            self._pad_sequence(value[start:stop], sequence_length)
                            for value in ordered
                        ),
                        valid,
                        *(value[start].clone() for value in batch[10:14]),
                    )
                )
        if not chunks:
            raise ValueError("at least one recurrent rollout is required")
        stacked = tuple(
            torch.stack([chunk[index] for chunk in chunks], dim=0)
            for index in range(8)
        )
        recurrent_state = RecurrentState(
            *(
                torch.stack([chunk[index] for chunk in chunks], dim=0)
                .unsqueeze(0)
                for index in range(8, 12)
            )
        )
        return (*stacked, recurrent_state)

    def _prepare_recurrent_batch(self, sequences):
        device = next(self.network.parameters()).device
        (
            states,
            actions,
            old_logprobs,
            advantages,
            returns,
            policy_masks,
            episode_starts,
            valid_masks,
            recurrent_state,
        ) = sequences
        tensors = (
            states,
            actions,
            old_logprobs,
            advantages,
            returns,
            policy_masks,
            episode_starts,
            valid_masks,
        )
        tensors = tuple(value.to(device) for value in tensors)
        recurrent_state = RecurrentState(
            *(value.to(device) for value in recurrent_state)
        )
        policy_valid = tensors[5] * tensors[7]
        selected_advantages = tensors[3][policy_valid > 0.5]
        if selected_advantages.numel() == 0:
            raise ValueError(
                "PPO update requires at least one policy sample"
            )
        advantage_std = selected_advantages.std(unbiased=False)
        if (
            torch.isfinite(advantage_std).item()
            and advantage_std.item() > 1e-8
        ):
            normalized = (
                tensors[3] - selected_advantages.mean()
            ) / (advantage_std + 1e-8)
        else:
            normalized = tensors[3] - selected_advantages.mean()
        tensors = (*tensors[:3], normalized, *tensors[4:])
        return (*tensors, recurrent_state)

    def _loss_for_recurrent_batch(
        self,
        states,
        actions,
        old_logprobs,
        advantages,
        returns,
        policy_masks,
        episode_starts,
        valid_masks,
        recurrent_state,
    ):
        distribution, values, _ = self.network.forward_recurrent(
            states,
            recurrent_state,
            episode_starts,
        )
        new_logprobs = self.network._squashed_log_prob(
            distribution,
            actions,
        )
        entropy = distribution.entropy().sum(dim=-1, keepdim=True)
        log_ratio = torch.clamp(
            new_logprobs - old_logprobs,
            min=-20.0,
            max=20.0,
        )
        ratio = torch.exp(log_ratio)
        policy_valid = policy_masks * valid_masks
        policy_count = policy_valid.sum().clamp_min(1.0)
        valid_count = valid_masks.sum().clamp_min(1.0)
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - self.eps, 1.0 + self.eps)
            * advantages
        )
        actor_loss = -(
            torch.min(surr1, surr2) * policy_valid
        ).sum() / policy_count
        critic_loss = (
            values.sub(returns).square() * valid_masks
        ).sum() / valid_count
        entropy_loss = (entropy * policy_valid).sum() / policy_count
        loss = (
            actor_loss
            + self.value_coef * critic_loss
            - self.entropy_coef * entropy_loss
        )
        approx_kl = (
            ((ratio - 1.0) - log_ratio) * policy_valid
        ).sum() / policy_count
        clip_fraction = (
            (torch.abs(ratio - 1.0) > self.eps).float() * policy_valid
        ).sum() / policy_count
        return (
            loss,
            actor_loss,
            critic_loss,
            entropy_loss,
            approx_kl,
            clip_fraction,
        )

    def _update_recurrent(self, sequences):
        prepared = self._prepare_recurrent_batch(sequences)
        sequence_count = prepared[0].shape[0]
        sequences_per_minibatch = max(
            1,
            self.minibatch_size // self.recurrent_sequence_length,
        )
        stats = self.empty_stats()
        for _ in range(self.ppo_epochs):
            indices = torch.randperm(
                sequence_count,
                device=prepared[0].device,
            )
            for start in range(0, sequence_count, sequences_per_minibatch):
                batch_indices = indices[start:start + sequences_per_minibatch]
                recurrent = RecurrentState(
                    *(value[:, batch_indices] for value in prepared[8])
                )
                loss_values = self._loss_for_recurrent_batch(
                    *(value[batch_indices] for value in prepared[:8]),
                    recurrent,
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
            _, _, updated_values = self.network.evaluate_action_sequences(
                prepared[0],
                prepared[1],
                prepared[8],
                prepared[6],
            )
            selected = prepared[7].squeeze(-1) > 0.5
            returns = prepared[4][selected]
            values = updated_values[selected]
            return_variance = torch.var(returns, unbiased=False)
            if return_variance.item() > 1e-8:
                residual_variance = torch.var(
                    returns - values,
                    unbiased=False,
                )
                stats["explained_variance"] = float(
                    (1.0 - residual_variance / return_variance).item()
                )
            else:
                stats["explained_variance"] = 0.0
        stats.update(self.network.diagnostic_stats(prepared[0][selected]))
        return stats

    def train_buffers(self, buffers):
        """Train on independent trajectories without leaking GAE across envs."""
        active_buffers = [buffer for buffer in buffers if len(buffer) > 0]
        if not active_buffers:
            return self.empty_stats()
        stats = self._update_recurrent(
            self._recurrent_sequences(active_buffers)
        )
        for buffer in active_buffers:
            buffer.clear()
        return stats
