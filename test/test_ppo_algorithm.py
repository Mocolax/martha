"""Focused algorithm tests for the PyTorch PPO implementation."""

import importlib
import math
from types import SimpleNamespace

import numpy as np
import pytest


torch = pytest.importorskip("torch")

train_module = importlib.import_module("martha.PPO.train")
evaluate_module = importlib.import_module("martha.PPO.evaluate")

from martha.PPO.buffer import RolloutBuffer  # noqa: E402
from martha.PPO.analytics import _load_metrics, _values  # noqa: E402
from martha.PPO.checkpoint import (  # noqa: E402
    load_policy,
    validate_checkpoint,
)
from martha.PPO.logic import PPOLogic  # noqa: E402
from martha.PPO.martha_env import (  # noqa: E402
    ACTION_SIZE,
    LASER_SECTORS,
    OBSERVATION_SIZE,
    POLICY_ARCHITECTURE,
    POLICY_CONTRACT_VERSION,
)
from martha.PPO.network import ActorCritic  # noqa: E402
from martha.PPO.observations import (  # noqa: E402
    OBSERVATION_FRAME_SIZE,
    OBSERVATION_HISTORY_FRAMES,
    OBSERVATION_HISTORY_SECONDS,
)
from martha.PPO.reward import RewardConfig  # noqa: E402
from martha.PPO.parallel_env import (  # noqa: E402
    ParallelWorkerConfig,
    RemoteMarthaEnv,
    _simulation_launch_command,
)
from martha.PPO.train import (  # noqa: E402
    DEFAULTS as TRAINING_DEFAULTS,
    METRIC_FIELDS,
    _cuda_rng_states_on_cpu,
    _evaluation_worlds,
    apply_entropy_schedule,
    entropy_coefficient_for_episode,
    _parallel_replacement_requests,
    _parallel_reset_batch,
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


def test_single_gazebo_environment_is_always_trainer_managed(monkeypatch):
    expected = object()
    args = SimpleNamespace(backend="gazebo", num_envs=1)
    monkeypatch.setattr(train_module, "train_gazebo", lambda received: expected)

    assert train_module.train(args) is expected


@pytest.mark.parametrize("gui", (False, True))
def test_parallel_worker_launches_only_its_requested_gui(gui):
    config = ParallelWorkerConfig(
        index=0,
        ros_domain_id=50,
        gazebo_port=11400,
        sim_speed_factor=4.0,
        gui=gui,
        startup_timeout=90.0,
        log_path="worker.log",
        environment_kwargs={},
    )

    command = _simulation_launch_command(config)

    assert f"gui:={'true' if gui else 'false'}" in command
    assert "sim_speed_factor:=4.0" in command


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


def test_navigation_encoder_matches_the_multibranch_architecture():
    extractor = ActorCritic().actor_feature_extractor
    first_convolution = extractor.laser_branch[0]
    second_convolution = extractor.laser_branch[2]
    laser_linear = extractor.laser_branch[5]

    assert first_convolution.in_channels == 4
    assert first_convolution.out_channels == 16
    assert first_convolution.kernel_size == (6,)
    assert first_convolution.stride == (3,)
    assert second_convolution.in_channels == 16
    assert second_convolution.out_channels == 32
    assert second_convolution.kernel_size == (5,)
    assert second_convolution.stride == (2,)
    assert laser_linear.in_features == 128
    assert laser_linear.out_features == 256
    assert extractor.orientation_branch[0].in_features == 8
    assert extractor.orientation_branch[0].out_features == 32
    assert extractor.distance_branch[0].in_features == 4
    assert extractor.distance_branch[0].out_features == 16
    assert extractor.velocity_branch[0].in_features == 12
    assert extractor.velocity_branch[0].out_features == 32
    assert extractor.fusion[0].in_features == 336
    assert extractor.fusion[0].out_features == 384


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
    environment = SimpleNamespace(predefined_maps=tuple(range(9)))

    assert args.eval_episodes == TRAINING_DEFAULTS.eval_episodes
    assert args.eval_max_steps == TRAINING_DEFAULTS.eval_max_steps
    assert args.reward_scale == pytest.approx(TRAINING_DEFAULTS.reward_scale)
    assert _evaluation_worlds(environment, args) == [0, 4, 8]

    args.eval_map_count = 0
    assert _evaluation_worlds(environment, args) == list(range(9))


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


def test_checkpoint_configuration_serializes_the_active_reward_config():
    reward_config = RewardConfig(wiggle_window_steps=7)
    environment = SimpleNamespace(
        observation_space=SimpleNamespace(shape=(OBSERVATION_SIZE,)),
        action_space=SimpleNamespace(shape=(3,)),
        policy_contract={"scan_range_max": 8.0},
        max_goal_distance=12.0,
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


def test_remote_environment_exposes_the_worker_reward_config():
    reward_config = RewardConfig(wiggle_window_steps=7)
    environment = RemoteMarthaEnv(
        connection=object(),
        process=object(),
        metadata={
            "observation_low": np.full(OBSERVATION_SIZE, -1.0),
            "observation_high": np.ones(OBSERVATION_SIZE),
            "action_low": np.full(ACTION_SIZE, -1.0),
            "action_high": np.ones(ACTION_SIZE),
            "action_limits": {
                "max_vx": 0.35,
                "max_vy": 0.35,
                "max_wz": 0.8,
                "max_action_delta": 0.35,
            },
            "reward_config": vars(reward_config),
            "policy_contract": {},
            "max_goal_distance": 12.0,
            "world_count": 9,
        },
    )

    assert environment.reward_config == reward_config


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
    )

    apply_entropy_schedule(logic, args, 1500)

    assert logic.entropy_coef == pytest.approx(0.001)


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


class _FakeRemoteEnvironment:
    def __init__(self, index, events, *, receive_error=None):
        self.index = index
        self.events = events
        self.receive_error = receive_error

    def send_reset(self, *, seed, options):
        self.events.append(("send", self.index, seed, options))

    def receive_reset(self):
        self.events.append(("receive", self.index))
        if self.receive_error is not None:
            raise self.receive_error
        return f"observation-{self.index}", {"worker": self.index}


def test_parallel_reset_batch_sends_every_reset_before_receiving():
    events = []
    environments = [
        _FakeRemoteEnvironment(index, events) for index in range(3)
    ]
    requests = [(index, 100 + index, {}) for index in range(3)]

    results = _parallel_reset_batch(environments, requests)

    assert [event[0] for event in events] == [
        "send",
        "send",
        "send",
        "receive",
        "receive",
        "receive",
    ]
    assert set(results) == {0, 1, 2}


def test_parallel_reset_batch_drains_responses_after_worker_error():
    events = []
    environments = [
        _FakeRemoteEnvironment(
            0,
            events,
            receive_error=RuntimeError("reset failed"),
        ),
        _FakeRemoteEnvironment(1, events),
    ]

    with pytest.raises(RuntimeError, match="worker\\(s\\): 0"):
        _parallel_reset_batch(
            environments,
            [(0, 100, {}), (1, 101, {})],
        )

    assert ("receive", 1) in events


def test_parallel_replacements_do_not_oversubscribe_episode_limit():
    args = SimpleNamespace(
        backend="gazebo",
        episodes=10,
        seed=42,
        map_index=None,
        episodes_per_map=20,
    )

    requests, next_episode = _parallel_replacement_requests(
        args,
        [3, 4, 5, 6],
        next_episode=9,
        world_count=9,
    )

    assert [request[0] for request in requests] == [3, 4]
    assert requests[0][1] != requests[1][1]
    assert requests[0][2]["world_index"] == requests[1][2]["world_index"]
    assert next_episode == 11


def test_training_map_blocks_are_stable_and_cover_a_random_cycle_once():
    args = SimpleNamespace(
        backend="gazebo",
        map_index=None,
        episodes_per_map=20,
        seed=42,
    )
    world_count = 9
    first_cycle = [
        training_world_index(args, world_count, 1 + 20 * block)
        for block in range(world_count)
    ]

    assert all(
        training_world_index(args, world_count, episode) == first_cycle[0]
        for episode in range(1, 21)
    )
    assert all(
        training_world_index(args, world_count, episode) == first_cycle[1]
        for episode in range(21, 41)
    )
    assert len(set(first_cycle)) == world_count


def test_training_map_blocks_are_seeded_and_resume_without_scheduler_state():
    common = {
        "backend": "gazebo",
        "map_index": None,
        "episodes_per_map": 20,
    }
    first = SimpleNamespace(seed=42, **common)
    repeat = SimpleNamespace(seed=42, **common)
    different = SimpleNamespace(seed=43, **common)
    world_count = 9
    episodes = range(1, 181)

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
    assert first_sequence[94:] == [
        training_world_index(first, world_count, episode)
        for episode in range(95, 181)
    ]


def test_training_map_index_override_disables_block_rotation():
    args = SimpleNamespace(
        backend="gazebo",
        map_index=4,
        episodes_per_map=20,
        seed=42,
    )

    assert training_world_index(args, 9, 1) == 4
    assert training_world_index(args, 9, 200) == 4


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
    contract = {
        "version": POLICY_CONTRACT_VERSION,
        "observation_size": OBSERVATION_SIZE,
        "action_size": ACTION_SIZE,
        "laser_sectors": LASER_SECTORS,
        "architecture": POLICY_ARCHITECTURE,
        "observation_layout": "frame_major",
        "observation_frame_size": OBSERVATION_FRAME_SIZE,
        "observation_history_frames": OBSERVATION_HISTORY_FRAMES,
        "observation_history_seconds": OBSERVATION_HISTORY_SECONDS,
        "scan_range_max": 8.0,
        "max_goal_distance": 12.0,
        "action_limits": {
            "max_vx": 0.35,
            "max_vy": 0.35,
            "max_wz": 0.80,
            "max_action_delta": 0.35,
        },
    }
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
    first_convolution = restored.actor_feature_extractor.laser_branch[0]
    assert first_convolution.in_channels == OBSERVATION_HISTORY_FRAMES
    assert first_convolution.kernel_size == (6,)
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
