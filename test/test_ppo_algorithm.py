"""Focused algorithm tests for the PyTorch PPO implementation."""

import math

import pytest


torch = pytest.importorskip("torch")

from martha.PPO.buffer import RolloutBuffer  # noqa: E402
from martha.PPO.logic import PPOLogic  # noqa: E402
from martha.PPO.network import ActorCritic  # noqa: E402


def _logic(gamma=0.90, lam=0.80):
    network = ActorCritic(state_dim=3, action_dim=2, hidden_dim=8)
    return PPOLogic(
        network,
        gamma=gamma,
        lam=lam,
        ppo_epochs=1,
        minibatch_size=2,
    )


def test_truncation_bootstraps_value_while_terminal_does_not():
    logic = _logic()
    rewards = torch.tensor([[1.0]])
    values = torch.tensor([[0.5]])
    next_values = torch.tensor([[2.0]])
    episode_ends = torch.tensor([[1.0]])

    terminal_advantage, terminal_return = logic.compute_gae(
        rewards=rewards,
        values=values,
        next_values=next_values,
        terminateds=torch.tensor([[1.0]]),
        episode_ends=episode_ends,
    )
    truncated_advantage, truncated_return = logic.compute_gae(
        rewards=rewards,
        values=values,
        next_values=next_values,
        terminateds=torch.tensor([[0.0]]),
        episode_ends=episode_ends,
    )

    assert terminal_advantage.item() == pytest.approx(0.5)
    assert terminal_return.item() == pytest.approx(1.0)
    assert truncated_advantage.item() == pytest.approx(2.3)
    assert truncated_return.item() == pytest.approx(2.8)


def test_episode_end_cuts_gae_trace_without_losing_truncation_bootstrap():
    logic = _logic(gamma=0.90, lam=0.95)
    advantages, returns = logic.compute_gae(
        rewards=torch.tensor([[0.0], [100.0]]),
        values=torch.zeros((2, 1)),
        next_values=torch.tensor([[2.0], [0.0]]),
        terminateds=torch.tensor([[0.0], [1.0]]),
        episode_ends=torch.tensor([[1.0], [1.0]]),
    )

    assert advantages[0].item() == pytest.approx(1.8)
    assert returns[0].item() == pytest.approx(1.8)
    assert advantages[1].item() == pytest.approx(100.0)
    assert returns[1].item() == pytest.approx(100.0)


def test_rollout_buffer_preserves_terminal_and_episode_end_masks():
    buffer = RolloutBuffer()
    buffer.store(
        state=torch.tensor([[1.0, 2.0, 3.0]]),
        action=torch.tensor([[0.1, -0.2]]),
        logprob=torch.tensor([[-0.3]]),
        reward=1.25,
        value=torch.tensor([[0.5]]),
        next_value=torch.tensor([[0.75]]),
        terminated=False,
        episode_end=True,
    )
    buffer.store(
        state=(4.0, 5.0, 6.0),
        action=(0.2, 0.3),
        logprob=-0.4,
        reward=-2.0,
        done=True,
        value=0.25,
    )

    (
        states,
        actions,
        logprobs,
        rewards,
        values,
        next_values,
        terminateds,
        episode_ends,
    ) = buffer.get_training_batch()

    assert len(buffer) == 2
    assert states.shape == (2, 3)
    assert actions.shape == (2, 2)
    for tensor in (
        logprobs,
        rewards,
        values,
        next_values,
        terminateds,
        episode_ends,
    ):
        assert tensor.shape == (2, 1)
        assert torch.isfinite(tensor).all()
    torch.testing.assert_close(next_values[:, 0], torch.tensor([0.75, 0.0]))
    torch.testing.assert_close(terminateds[:, 0], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(episode_ends[:, 0], torch.tensor([1.0, 1.0]))

    buffer.clear()
    assert len(buffer) == 0


def test_actor_critic_action_logprob_and_value_shapes_are_finite():
    torch.manual_seed(7)
    network = ActorCritic(state_dim=5, action_dim=3, hidden_dim=16)
    state = torch.linspace(-1.0, 1.0, 5)

    action, logprob, value = network.get_action(state)

    assert action.shape == (3,)
    assert logprob.shape == (1, 1)
    assert value.shape == (1, 1)
    assert torch.isfinite(action).all()
    assert torch.isfinite(logprob).all()
    assert torch.isfinite(value).all()
    assert torch.all(action >= -1.0)
    assert torch.all(action <= 1.0)

    states = torch.stack((state, -state))
    actions = torch.stack((action, -action))
    logprobs, entropy, values = network.evaluate_actions(states, actions)
    assert logprobs.shape == entropy.shape == values.shape == (2, 1)
    assert torch.isfinite(logprobs).all()
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(values).all()


def test_tiny_ppo_update_is_finite_and_changes_parameters():
    torch.manual_seed(11)
    network = ActorCritic(state_dim=5, action_dim=3, hidden_dim=16)
    logic = PPOLogic(
        network,
        lr=1e-3,
        ppo_epochs=2,
        minibatch_size=4,
    )
    states = torch.randn((8, 5))
    samples = [network.get_action(state) for state in states]
    actions = torch.stack([sample[0] for sample in samples])
    old_logprobs = torch.cat([sample[1] for sample in samples])
    values = torch.cat([sample[2] for sample in samples])
    advantages = torch.linspace(-1.0, 1.0, 8).view(-1, 1)
    returns = values + advantages
    before = {
        name: parameter.detach().clone()
        for name, parameter in network.named_parameters()
    }

    stats = logic.update(
        states,
        actions,
        old_logprobs,
        advantages,
        returns,
    )

    assert stats["updates"] == 4
    for key in ("loss", "actor_loss", "critic_loss", "entropy"):
        assert math.isfinite(stats[key])
    assert any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in network.named_parameters()
    )


def test_state_dict_round_trip_preserves_deterministic_policy(tmp_path):
    torch.manual_seed(23)
    original = ActorCritic(state_dim=5, action_dim=3, hidden_dim=16)
    checkpoint_path = tmp_path / "policy_state.pt"
    torch.save(original.state_dict(), checkpoint_path)

    restored = ActorCritic(state_dim=5, action_dim=3, hidden_dim=16)
    restored.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    state = torch.linspace(-0.5, 0.5, 5)
    original_action, original_logprob, original_value = original.get_action(
        state,
        deterministic=True,
    )
    restored_action, restored_logprob, restored_value = restored.get_action(
        state,
        deterministic=True,
    )

    torch.testing.assert_close(restored_action, original_action)
    torch.testing.assert_close(restored_logprob, original_logprob)
    torch.testing.assert_close(restored_value, original_value)
