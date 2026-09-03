"""Focused algorithm tests for the PyTorch PPO implementation."""

import importlib
import math
from types import SimpleNamespace

import numpy as np
import pytest


torch = pytest.importorskip("torch")

train_module = importlib.import_module("martha.PPO.train")
evaluate_module = importlib.import_module("martha.PPO.evaluate")

from martha.PPO.actions import ActionLimits  # noqa: E402
from martha.PPO.buffer import RolloutBuffer  # noqa: E402
from martha.PPO.analytics import _load_metrics, _values  # noqa: E402
from martha.PPO.checkpoint import (  # noqa: E402
    load_policy,
    validate_checkpoint,
)
from martha.PPO.logic import PPOLogic  # noqa: E402
from martha.PPO.martha_env import (  # noqa: E402
    ACTION_SIZE,
    OBSERVATION_SIZE,
    POLICY_CONTRACT_VERSION,
    build_policy_contract,
)
from martha.PPO.network import (  # noqa: E402
    ActorCritic,
    NavigationFeatureExtractor,
    POLICY_ARCHITECTURE,
    RECURRENT_HIDDEN_SIZE,
)
from martha.PPO.observations import (  # noqa: E402
    OBSERVATION_FRAME_SIZE,
    OBSERVATION_HISTORY_FRAMES,
)
from martha.PPO.reward import RewardConfig  # noqa: E402
from martha.PPO.train import (  # noqa: E402
    DEFAULTS as TRAINING_DEFAULTS,
    METRIC_FIELDS,
    _cuda_rng_states_on_cpu,
    _evaluation_worlds,
    _truncate_active_for_wall_limit,
    apply_entropy_schedule,
    curriculum_max_goal_distance,
    entropy_coefficient_for_episode,
    _validate_resume_reward_scale,
    parse_args,
    scale_reward_for_ppo,
    training_world_index,
    write_metric,
)


def _logic(gamma=0.90, lam=0.80):
    network = ActorCritic()
    return PPOLogic(
        network,
        gamma=gamma,
        lam=lam,
        ppo_epochs=1,
        minibatch_size=2,
    )


def test_feedforward_batch_normalizes_policy_gae_advantages():
    logic = _logic()
    batch_size = 3
    advantages = torch.tensor([[-3.0], [1.0], [7.0]])

    prepared = logic._prepare_batch(
        torch.zeros((batch_size, OBSERVATION_SIZE)),
        torch.zeros((batch_size, ACTION_SIZE)),
        torch.zeros((batch_size, 1)),
        advantages.clone(),
        torch.zeros((batch_size, 1)),
        torch.tensor([[1.0], [0.0], [1.0]]),
    )

    selected = advantages[torch.tensor([True, False, True])]
    expected = (advantages - selected.mean()) / selected.std(unbiased=False)
    torch.testing.assert_close(prepared[3], expected)


def test_recurrent_batch_normalizes_valid_policy_gae_advantages():
    logic = _logic()
    network = logic.network
    batch_size = 2
    sequence_length = 3
    advantages = torch.tensor(
        [[[-4.0], [0.5], [8.0]], [[2.0], [-1.5], [6.0]]]
    )

    prepared = logic._prepare_recurrent_batch(
        (
            torch.zeros((batch_size, sequence_length, OBSERVATION_SIZE)),
            torch.zeros((batch_size, sequence_length, ACTION_SIZE)),
            torch.zeros((batch_size, sequence_length, 1)),
            advantages.clone(),
            torch.zeros((batch_size, sequence_length, 1)),
            torch.tensor(
                [[[1.0], [1.0], [0.0]], [[1.0], [0.0], [1.0]]]
            ),
            torch.zeros((batch_size, sequence_length, 1)),
            torch.tensor(
                [[[1.0], [1.0], [1.0]], [[1.0], [1.0], [0.0]]]
            ),
            network.initial_recurrent_state(batch_size),
        )
    )

    policy_valid = torch.tensor(
        [[[1.0], [1.0], [0.0]], [[1.0], [0.0], [0.0]]]
    )
    selected = advantages[policy_valid > 0.5]
    expected = (advantages - selected.mean()) / selected.std(unbiased=False)
    torch.testing.assert_close(prepared[3], expected)


def test_single_gazebo_environment_is_always_trainer_managed(monkeypatch):
    expected = object()
    args = SimpleNamespace(backend="gazebo", num_envs=1)
    monkeypatch.setattr(train_module, "train_gazebo", lambda received: expected)

    assert train_module.train(args) is expected


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
        value=0.25,
        next_value=0.0,
        terminated=True,
        episode_end=True,
        policy_sample=False,
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
        policy_masks,
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
        policy_masks,
    ):
        assert tensor.shape == (2, 1)
        assert torch.isfinite(tensor).all()
    torch.testing.assert_close(next_values[:, 0], torch.tensor([0.75, 0.0]))
    torch.testing.assert_close(terminateds[:, 0], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(episode_ends[:, 0], torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(policy_masks[:, 0], torch.tensor([1.0, 0.0]))

    buffer.clear()
    assert len(buffer) == 0


def test_supervised_stop_samples_train_critic_but_are_masked_from_actor_loss():
    logic = _logic()
    states = torch.randn((2, OBSERVATION_SIZE))
    actions = torch.zeros((2, ACTION_SIZE))
    with torch.no_grad():
        old_logprobs, _, values = logic.network.evaluate_actions(states, actions)
    advantages = torch.ones((2, 1))
    returns = values + advantages

    losses = logic._loss_for_batch(
        states,
        actions,
        old_logprobs,
        advantages,
        returns,
        torch.tensor([[1.0], [0.0]]),
    )

    assert losses[1].item() == pytest.approx(-1.0)
    assert losses[2].item() > 0.0


def test_actor_critic_action_logprob_and_value_shapes_are_finite():
    torch.manual_seed(7)
    network = ActorCritic()
    state = torch.linspace(-1.0, 1.0, OBSERVATION_SIZE)

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


def test_actor_critic_vectorized_actions_preserve_environment_batch():
    network = ActorCritic()
    states = torch.randn((4, OBSERVATION_SIZE))

    actions, logprobs, values = network.get_actions(states)

    assert actions.shape == (4, 3)
    assert logprobs.shape == (4, 1)
    assert values.shape == (4, 1)
    assert torch.isfinite(actions).all()
    assert torch.isfinite(logprobs).all()
    assert torch.isfinite(values).all()


def test_lstm_memory_advances_and_episode_start_resets_it():
    torch.manual_seed(19)
    network = ActorCritic()
    observation = torch.randn(OBSERVATION_SIZE)
    initial = network.initial_recurrent_state(1)

    first_action, _, _, advanced = network.get_actions_recurrent(
        observation,
        initial,
        episode_starts=[True],
        deterministic=True,
    )
    second_action, _, _, _ = network.get_actions_recurrent(
        observation,
        advanced,
        episode_starts=[False],
        deterministic=True,
    )
    reset_action, _, _, reset_advanced = network.get_actions_recurrent(
        observation,
        advanced,
        episode_starts=[True],
        deterministic=True,
    )

    assert any(torch.count_nonzero(value) > 0 for value in advanced)
    assert not torch.equal(second_action, first_action)
    torch.testing.assert_close(reset_action, first_action)
    for reset_value, first_value in zip(reset_advanced, advanced):
        torch.testing.assert_close(reset_value, first_value)


def test_batched_lstm_sequence_matches_stepwise_execution():
    torch.manual_seed(29)
    network = ActorCritic()
    states = torch.randn((2, 5, OBSERVATION_SIZE))
    starts = torch.zeros((2, 5, 1))
    starts[:, 0] = 1.0
    starts[1, 3] = 1.0
    initial = network.initial_recurrent_state(2)

    sequence_dist, sequence_values, sequence_end = (
        network.forward_recurrent(states, initial, starts)
    )
    step_state = initial
    step_means = []
    step_values = []
    for step in range(states.shape[1]):
        dist, values, step_state = network.forward_recurrent(
            states[:, step],
            step_state,
            starts[:, step:step + 1],
        )
        step_means.append(dist.mean)
        step_values.append(values)

    torch.testing.assert_close(
        sequence_dist.mean,
        torch.cat(step_means, dim=1),
    )
    torch.testing.assert_close(
        sequence_values,
        torch.cat(step_values, dim=1),
    )
    for sequence_value, step_value in zip(sequence_end, step_state):
        torch.testing.assert_close(sequence_value, step_value)


def test_recurrent_rollout_chunks_keep_resets_padding_and_initial_memory():
    buffers = [RolloutBuffer(), RolloutBuffer()]
    network = ActorCritic()
    recurrent = network.initial_recurrent_state(1)
    for step in range(3):
        buffers[0].store(
            state=np.full(OBSERVATION_SIZE, step, dtype=np.float32),
            action=np.zeros(ACTION_SIZE, dtype=np.float32),
            logprob=0.0,
            reward=0.0,
            value=0.0,
            next_value=0.0,
            terminated=False,
            episode_end=step == 1,
            recurrent_state=recurrent,
        )
    buffers[1].store(
        state=np.zeros(OBSERVATION_SIZE, dtype=np.float32),
        action=np.zeros(ACTION_SIZE, dtype=np.float32),
        logprob=0.0,
        reward=0.0,
        value=0.0,
        next_value=0.0,
        terminated=False,
        episode_end=False,
        recurrent_state=recurrent,
    )
    logic = PPOLogic(
        network,
        ppo_epochs=1,
        minibatch_size=4,
        recurrent_sequence_length=4,
    )

    sequences = logic._recurrent_sequences(buffers)

    assert sequences[0].shape == (2, 4, OBSERVATION_SIZE)
    assert sequences[7].sum().item() == pytest.approx(4.0)
    torch.testing.assert_close(
        sequences[6][0, :, 0],
        torch.tensor([1.0, 0.0, 1.0, 0.0]),
    )
    assert sequences[8].actor_h.shape == (
        1,
        2,
        RECURRENT_HIDDEN_SIZE,
    )


def test_collected_recurrent_rollout_updates_lstm_parameters():
    torch.manual_seed(31)
    network = ActorCritic()
    logic = PPOLogic(
        network,
        lr=1e-3,
        ppo_epochs=1,
        minibatch_size=8,
        recurrent_sequence_length=4,
    )
    buffer = RolloutBuffer()
    recurrent = network.initial_recurrent_state(1)
    observations = torch.randn((9, OBSERVATION_SIZE))
    for step in range(8):
        transition_state = recurrent
        action, logprob, value, recurrent = network.get_actions_recurrent(
            observations[step],
            recurrent,
            episode_starts=[step == 0],
        )
        terminal = step == 7
        next_value = (
            torch.zeros((1, 1))
            if terminal
            else network.get_value_recurrent(
                observations[step + 1],
                recurrent,
            )
        )
        buffer.store(
            state=observations[step],
            action=action,
            logprob=logprob,
            reward=float(step % 3) - 0.5,
            value=value,
            next_value=next_value,
            terminated=terminal,
            episode_end=terminal,
            recurrent_state=transition_state,
            episode_start=step == 0,
        )
    before = {
        name: parameter.detach().clone()
        for name, parameter in network.actor_lstm.named_parameters()
    }

    stats = logic.train_buffer(buffer)

    assert stats["updates"] == 1
    assert all(math.isfinite(float(value)) for value in stats.values())
    assert any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in network.actor_lstm.named_parameters()
    )
    assert len(buffer) == 0


def test_navigation_encoder_consumes_one_frame_with_circular_laser_convs():
    extractor = NavigationFeatureExtractor()
    first_convolution = extractor.laser_branch[0][0]
    second_convolution = extractor.laser_branch[0][2]
    laser_linear = extractor.laser_branch[1]

    assert first_convolution.in_channels == 1
    assert first_convolution.out_channels == 16
    assert first_convolution.kernel_size == (6,)
    assert first_convolution.stride == (3,)
    assert first_convolution.padding_mode == "circular"
    assert second_convolution.in_channels == 16
    assert second_convolution.out_channels == 32
    assert second_convolution.kernel_size == (5,)
    assert second_convolution.stride == (2,)
    assert second_convolution.padding_mode == "circular"
    assert laser_linear.out_features == 256
    assert extractor.orientation_branch[0].in_features == 2
    assert extractor.orientation_branch[0].out_features == 32
    assert extractor.distance_branch[0].in_features == 1
    assert extractor.distance_branch[0].out_features == 16
    assert extractor.velocity_branch[0].in_features == 3
    assert extractor.velocity_branch[0].out_features == 32
    assert extractor.fusion[0].in_features == 336
    assert extractor.fusion[0].out_features == 384


def test_observation_carries_exactly_one_frame():
    assert OBSERVATION_HISTORY_FRAMES == 1
    assert OBSERVATION_SIZE == OBSERVATION_FRAME_SIZE


def test_policy_mean_head_starts_near_zero():
    network = ActorCritic()

    # The 0.01 orthogonal gain must keep the initial policy small and centred.
    assert network.actor_mean.weight.abs().max().item() < 0.05
    assert network.actor_mean.bias.abs().max().item() == 0.0


def test_default_network_architecture_is_the_recurrent_multibranch_encoder():
    ActorCritic()

    assert POLICY_ARCHITECTURE == "multibranch_lstm"


def test_actor_and_critic_have_disjoint_parameters_and_gradients():
    network = ActorCritic()
    actor_parameters = list(network.actor_parameters())
    critic_parameters = list(network.critic_parameters())

    assert {id(value) for value in actor_parameters}.isdisjoint(
        {id(value) for value in critic_parameters}
    )
    states = torch.randn((8, OBSERVATION_SIZE))
    _, values = network(states)
    values.square().mean().backward()

    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in actor_parameters
    )
    assert any(
        parameter.grad is not None
        and torch.count_nonzero(parameter.grad) > 0
        for parameter in critic_parameters
    )


def test_network_diagnostics_are_finite_and_bounded():
    network = ActorCritic()

    diagnostics = network.diagnostic_stats(
        torch.randn((32, OBSERVATION_SIZE))
    )

    assert diagnostics["policy_std"] > 0.0
    for name in ("actor_inactive_relu", "critic_inactive_relu"):
        assert 0.0 <= diagnostics[name] <= 1.0
        assert math.isfinite(diagnostics[name])


def test_periodic_evaluation_uses_editable_training_defaults():
    args = parse_args([])
    environment = SimpleNamespace(predefined_maps=tuple(range(6)))

    assert args.eval_episodes == TRAINING_DEFAULTS.eval_episodes
    assert args.eval_max_steps == TRAINING_DEFAULTS.eval_max_steps
    assert args.reward_scale == pytest.approx(TRAINING_DEFAULTS.reward_scale)
    assert args.physics_step_size == pytest.approx(0.002)
    assert args.lidar_samples == 180
    assert args.training_kinematic is True
    assert args.map_batch_episodes == TRAINING_DEFAULTS.map_batch_episodes
    assert args.max_wall_time_hours == pytest.approx(24.0)
    assert args.eval_every == TRAINING_DEFAULTS.eval_every
    assert args.eval_episodes == TRAINING_DEFAULTS.eval_episodes
    assert args.max_vx == pytest.approx(TRAINING_DEFAULTS.max_vx)
    assert args.max_vy == pytest.approx(TRAINING_DEFAULTS.max_vy)
    assert args.max_wz == pytest.approx(TRAINING_DEFAULTS.max_wz)
    assert args.max_action_delta == pytest.approx(0.35)
    assert _evaluation_worlds(environment, args) == list(range(6))

    args.eval_map_count = 0
    assert _evaluation_worlds(environment, args) == list(range(6))


def test_training_cli_only_exposes_resume():
    parser = train_module.build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert option_strings == {"-h", "--help", "--resume"}


def test_evaluation_cli_only_exposes_checkpoint():
    parser = evaluate_module.build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert option_strings == {"-h", "--help", "--checkpoint"}


def test_training_metrics_reject_an_old_csv_schema(tmp_path):
    metrics_path = tmp_path / "metrics.csv"
    metrics_path.write_text("episode,episode_reward\n1,-1.0\n", encoding="utf-8")
    current_row = {name: 0.0 for name in METRIC_FIELDS}

    with pytest.raises(ValueError, match="current metric schema"):
        write_metric(metrics_path, current_row)


def test_paper_reward_metrics_and_legacy_component_reports_are_compatible(tmp_path):
    assert {
        "reward_step",
        "reward_distance",
        "reward_orientation",
        "reward_shortest_distance",
        "reward_laser",
        "reward_wiggle",
        "reward_terminal",
    }.issubset(METRIC_FIELDS)

    metrics_path = tmp_path / "legacy_metrics.csv"
    metrics_path.write_text("episode,episode_reward\n1,-1.0\n", encoding="utf-8")
    metrics = _load_metrics(metrics_path)
    assert np.isnan(_values(metrics, "reward_distance")).all()


def test_reward_config_is_restored_from_checkpoint_and_legacy_uses_defaults():
    config = RewardConfig(wiggle_window_steps=7, laser_clearance_distance=0.7)
    checkpoint = {"config": {"reward_config": vars(config)}}

    restored = train_module.reward_config_from_checkpoint(checkpoint)

    assert restored == config
    assert train_module.reward_config_from_checkpoint({"config": {}}) == RewardConfig()


def test_resume_rejects_checkpoint_from_previous_reward_schema():
    saved = vars(RewardConfig()).copy()
    saved.pop("timeout_penalty")

    with pytest.raises(ValueError, match="start a new run"):
        train_module.reward_config_from_checkpoint(
            {"config": {"reward_config": saved}}
        )


def test_checkpoint_configuration_serializes_the_active_reward_config():
    reward_config = RewardConfig(wiggle_window_steps=7)
    environment = SimpleNamespace(
        observation_space=SimpleNamespace(shape=(OBSERVATION_SIZE,)),
        action_space=SimpleNamespace(shape=(3,)),
        policy_contract={"scan_range_max": 8.0},
        goal_distance_scale=3.0,
        action_limits=SimpleNamespace(
            max_vx=0.35,
            max_vy=0.35,
            max_wz=0.8,
            max_action_delta=0.35,
        ),
        reward_config=reward_config,
    )
    args = SimpleNamespace()

    config = train_module._serializable_config(args, environment)

    assert config["reward_config"] == vars(reward_config)


def test_reward_scaling_preserves_raw_reward_for_reporting():
    raw_reward = -120.5

    scaled_reward = scale_reward_for_ppo(raw_reward, 0.01)

    assert scaled_reward == pytest.approx(-1.205)
    assert raw_reward == pytest.approx(-120.5)


def test_entropy_schedule_has_exploration_consolidation_and_refinement_phases():
    coefficient = 0.002

    assert entropy_coefficient_for_episode(
        coefficient, 2000, 0.60, 0.30, 1
    ) == pytest.approx(coefficient)
    assert entropy_coefficient_for_episode(
        coefficient, 2000, 0.60, 0.30, 1200
    ) == pytest.approx(coefficient)
    assert entropy_coefficient_for_episode(
        coefficient, 2000, 0.60, 0.30, 1500
    ) == pytest.approx(0.001)
    assert entropy_coefficient_for_episode(
        coefficient, 2000, 0.60, 0.30, 1800
    ) == pytest.approx(0.0)
    assert entropy_coefficient_for_episode(
        coefficient, 2000, 0.60, 0.30, 2000
    ) == pytest.approx(0.0)
    # The same fractions scale the phase boundaries for a shorter run.
    assert entropy_coefficient_for_episode(
        coefficient, 1000, 0.60, 0.30, 750
    ) == pytest.approx(0.001)


def test_entropy_schedule_updates_the_live_ppo_weight():
    logic = _logic()
    args = SimpleNamespace(
        entropy_coef=0.002,
        episodes=2000,
        entropy_exploration_fraction=0.60,
        entropy_decay_fraction=0.30,
        policy_std_initial=0.40,
        policy_std_final=0.15,
    )

    apply_entropy_schedule(logic, args, 1500)

    assert logic.entropy_coef == pytest.approx(0.001)
    assert torch.exp(logic.network.actor_logstd).max().item() == pytest.approx(
        0.275
    )


def test_resume_requires_the_checkpoint_reward_scale():
    checkpoint = {"config": {"reward_scale": 0.01}}

    _validate_resume_reward_scale(
        SimpleNamespace(reward_scale=0.01),
        checkpoint,
    )
    with pytest.raises(ValueError, match="must match"):
        _validate_resume_reward_scale(
            SimpleNamespace(reward_scale=0.1),
            checkpoint,
        )


def test_training_rounds_cover_one_seeded_random_cycle_once():
    args = SimpleNamespace(
        backend="gazebo",
        map_index=None,
        num_envs=8,
        seed=42,
    )
    world_count = 6
    first_cycle = [
        training_world_index(args, world_count, 1 + 8 * block)
        for block in range(world_count)
    ]

    assert all(
        training_world_index(args, world_count, episode) == first_cycle[0]
        for episode in range(1, 9)
    )
    assert all(
        training_world_index(args, world_count, episode) == first_cycle[1]
        for episode in range(9, 17)
    )
    assert len(set(first_cycle)) == world_count


def test_training_map_blocks_are_seeded_and_resume_without_scheduler_state():
    common = {
        "backend": "gazebo",
        "map_index": None,
        "num_envs": 8,
    }
    first = SimpleNamespace(seed=42, **common)
    repeat = SimpleNamespace(seed=42, **common)
    different = SimpleNamespace(seed=43, **common)
    world_count = 6
    episodes = range(1, 97)

    first_sequence = [
        training_world_index(first, world_count, episode)
        for episode in episodes
    ]
    assert first_sequence == [
        training_world_index(repeat, world_count, episode)
        for episode in episodes
    ]
    assert first_sequence != [
        training_world_index(different, world_count, episode)
        for episode in episodes
    ]
    assert first_sequence[37:] == [
        training_world_index(first, world_count, episode)
        for episode in range(38, 97)
    ]


def test_training_map_index_override_disables_block_rotation():
    args = SimpleNamespace(
        backend="gazebo",
        map_index=4,
        num_envs=8,
        seed=42,
    )

    assert training_world_index(args, 6, 1) == 4
    assert training_world_index(args, 6, 200) == 4


def test_navigation_curriculum_balances_maps_and_expands_route_distances():
    args = SimpleNamespace(
        backend="gazebo",
        map_index=None,
        map_batch_episodes=12,
        seed=42,
        episodes=8000,
        curriculum_enabled=True,
        curriculum_easy_fraction=0.40,
        curriculum_medium_fraction=0.60,
        curriculum_full_fraction=0.85,
        curriculum_easy_max_distance=6.0,
        curriculum_medium_max_distance=10.0,
        curriculum_hard_max_distance=18.0,
    )

    early_worlds = {
        training_world_index(args, 6, episode)
        for episode in range(1, 73, 12)
    }
    assert early_worlds == set(range(6))
    assert curriculum_max_goal_distance(args, 1) == pytest.approx(6.0)
    assert curriculum_max_goal_distance(args, 3201) == pytest.approx(10.0)
    assert curriculum_max_goal_distance(args, 4801) == pytest.approx(18.0)
    assert curriculum_max_goal_distance(args, 6801) is None


def test_training_map_batch_size_enables_recycling_without_scheduler_state():
    args = SimpleNamespace(
        backend="gazebo",
        map_index=None,
        num_envs=4,
        map_batch_episodes=24,
        seed=42,
    )

    first = training_world_index(args, 6, 1)
    second = training_world_index(args, 6, 25)

    assert all(training_world_index(args, 6, episode) == first for episode in range(1, 25))
    assert second != first


def test_hard_wall_limit_marks_active_buffers_as_bootstrapped_truncations():
    buffers = [RolloutBuffer(), RolloutBuffer()]
    for buffer in buffers:
        buffer.store(
            state=np.zeros(OBSERVATION_SIZE),
            action=np.zeros(ACTION_SIZE),
            logprob=0.0,
            reward=0.0,
            value=0.0,
            next_value=0.5,
            terminated=False,
            episode_end=False,
        )
    active = {
        index: {
            "episode": index + 1,
            "last_info": {"position": (0.0, 0.0, 0.0)},
        }
        for index in range(2)
    }

    finished = _truncate_active_for_wall_limit(active, buffers)

    assert active == {}
    assert set(finished) == {0, 1}
    assert all(state["truncated"] for state in finished.values())
    assert all(state["info"]["wall_time_limit"] for state in finished.values())
    assert all(buffer.episode_ends[-1] == 1.0 for buffer in buffers)


def test_tiny_ppo_update_is_finite_and_changes_parameters():
    torch.manual_seed(11)
    network = ActorCritic()
    logic = PPOLogic(
        network,
        lr=1e-3,
        ppo_epochs=2,
        minibatch_size=4,
    )
    states = torch.randn((8, OBSERVATION_SIZE))
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
    for key in (
        "loss",
        "actor_loss",
        "critic_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "explained_variance",
        "policy_std",
        "actor_inactive_relu",
        "critic_inactive_relu",
    ):
        assert math.isfinite(stats[key])
    assert stats["approx_kl"] >= 0.0
    assert 0.0 <= stats["clip_fraction"] <= 1.0
    assert 0.0 <= stats["actor_inactive_relu"] <= 1.0
    assert 0.0 <= stats["critic_inactive_relu"] <= 1.0
    assert any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in network.named_parameters()
    )


def test_parallel_buffers_are_trained_and_cleared_as_independent_traces():
    buffers = [RolloutBuffer(), RolloutBuffer()]
    for index, buffer in enumerate(buffers):
        buffer.store(
            state=np.full(OBSERVATION_SIZE, float(index), dtype=np.float32),
            action=(0.1, -0.1, 0.0),
            logprob=-0.5,
            reward=float(index + 1),
            value=0.25,
            next_value=0.5,
            terminated=False,
            episode_end=False,
        )
    logic = _logic()

    stats = logic.train_buffers(buffers)

    assert stats["updates"] > 0
    assert all(len(buffer) == 0 for buffer in buffers)


def test_state_dict_round_trip_preserves_deterministic_policy(tmp_path):
    torch.manual_seed(23)
    original = ActorCritic()
    checkpoint_path = tmp_path / "policy_state.pt"
    torch.save(original.state_dict(), checkpoint_path)

    restored = ActorCritic()
    restored.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    state = torch.linspace(-0.5, 0.5, OBSERVATION_SIZE)
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


def test_checkpoint_requires_the_canonical_policy_contract(tmp_path):
    network = ActorCritic()
    contract = build_policy_contract(
        scan_range_max=8.0,
        goal_distance_scale=3.0,
        action_limits=ActionLimits(
            max_vx=0.35,
            max_vy=0.35,
            max_wz=0.80,
            max_action_delta=0.35,
        ),
    )
    assert contract["recurrent_hidden_size"] == RECURRENT_HIDDEN_SIZE
    assert contract["recurrent_hidden_size"] == network.actor_lstm.hidden_size
    checkpoint = {
        "model_state_dict": network.state_dict(),
        "policy_contract": contract,
        "config": {},
    }
    checkpoint_path = tmp_path / "canonical.pt"
    torch.save(checkpoint, checkpoint_path)

    validate_checkpoint(checkpoint)
    restored, loaded, limits = load_policy(
        checkpoint_path,
        torch.device("cpu"),
        expected_contract=contract,
    )

    assert loaded["policy_contract"] == contract
    assert limits.max_vx == pytest.approx(0.35)
    first_convolution = restored.actor_feature_extractor.laser_branch[0][0]
    assert first_convolution.in_channels == 1
    assert first_convolution.kernel_size == (6,)
    assert restored.actor_lstm.hidden_size == RECURRENT_HIDDEN_SIZE
    incompatible_contract = dict(contract)
    incompatible_contract["architecture"] = "some_other_architecture"
    with pytest.raises(ValueError, match="architecture mismatch"):
        validate_checkpoint(
            {
                **checkpoint,
                "policy_contract": incompatible_contract,
            }
        )
    with pytest.raises(ValueError, match="missing policy_contract"):
        validate_checkpoint(
            {
                "model_state_dict": network.state_dict(),
                "policy_contract_version": POLICY_CONTRACT_VERSION,
                "action_limits": contract["action_limits"],
            }
        )


def test_cuda_rng_checkpoint_states_are_normalized_to_cpu_byte_tensors():
    states = [torch.arange(32, dtype=torch.uint8)]

    normalized = _cuda_rng_states_on_cpu(states)

    assert len(normalized) == 1
    assert normalized[0].device.type == "cpu"
    assert normalized[0].dtype == torch.uint8
    torch.testing.assert_close(normalized[0], states[0])


@pytest.mark.parametrize("invalid", (None, torch.ones(2), [torch.ones(2)]))
def test_invalid_cuda_rng_checkpoint_states_are_rejected(invalid):
    with pytest.raises(ValueError, match="CUDA RNG state"):
        _cuda_rng_states_on_cpu(invalid)
