"""
Paper-derived reward shaping shared by PPO training and evaluation.

The equations implemented here are adapted from Jestel et al., *Obtaining
Robust Control and Navigation Policies for Multi-Robot Navigation via Deep
Reinforcement Learning* (2022), section 2.1.  The environment owns the small
per-episode :class:`RewardState`; this module keeps the reward calculation
pure and therefore straightforward to test.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


REWARD_COMPONENT_NAMES = (
    "step",
    "distance",
    "orientation",
    "shortest_distance",
    "laser",
    "velocity",
    "hold",
    "wiggle",
    "terminal",
)


@dataclass(frozen=True)
class RewardConfig:
    """Editable constants for the complete reward function."""

    # A small living cost makes finishing preferable to exhausting max_steps.
    # NOTE (anti-circling): at 0.0002 this is almost free -- orbiting for the
    # full 200-step stagnation window costs only ~0.04. If circling / stagnation
    # persists after the goal-angle encoding, raise this to ~0.002-0.003 to put
    # real time pressure on wandering. Bumping it needs a fresh run.
    step_penalty: float = 0.0002
    # Equation (4): distance progress.  The two scales are equal so the term
    # is potential-based: any approach/retreat cycle sums to zero and only net
    # progress toward the goal pays.  An asymmetric retreat let the agent farm
    # positive reward by oscillating in place until the timeout.
    distance_positive_scale: float = 0.05
    distance_negative_scale: float = 0.05
    # Equation (7): orientation to the goal.  Looking away gets no penalty;
    # the retained field keeps recently saved paper-reward checkpoints loadable.
    orientation_positive_scale: float = 0.0001
    orientation_negative_scale: float = 0.0
    # Equation (8): a new best distance within the current episode.
    shortest_distance_scale: float = 0.20
    # Dense clearance penalty: a sigmoid of the nearest-obstacle distance,
    # -laser_penalty_scale * sigmoid(steepness * (midpoint - minimum_scan)).
    # It is half strength at laser_penalty_midpoint and saturates near contact,
    # and -- unlike a hard cutoff -- keeps a gradient out to ~1.5 m. So a
    # corridor's two walls form a valley the policy can center in ("rails")
    # from the distance reward alone. This is the main obstacle signal, since
    # entering the band is no longer terminal; only a bumper contact ends it.
    laser_penalty_scale: float = 0.06
    laser_penalty_midpoint: float = 0.45
    laser_penalty_steepness: float = 6.0
    # Approach braking: penalize linear speed only near the goal, so the robot
    # arrives slowly and can hold still instead of overshooting. The penalty is
    # -velocity_penalty_scale * speed**2 * sigmoid(steepness * (slow - dist)),
    # gated so it is ~0 beyond velocity_slow_distance (fast travel stays free)
    # and full inside it. speed**2 makes a slow creep almost free.
    velocity_penalty_scale: float = 0.08
    velocity_slow_distance: float = 1.0
    velocity_penalty_steepness: float = 4.0
    # Optional "hold at the goal" shaping, OFF by default (both 0.0) so the goal
    # is learned purely from the terminal reward + the dwell. Set a small
    # goal_hold_bonus (e.g. 0.3/step while inside the tolerance) to give dense
    # feedback for staying, and a larger goal_leave_penalty (charged the step it
    # steps back out) so bouncing in/out to farm the per-step bonus never pays.
    goal_hold_bonus: float = 0.0
    goal_leave_penalty: float = 0.0
    # Equations (10)-(12): excessive direct left/right reversals.
    wiggle_penalty: float = 0.00002
    wiggle_turn_rate_threshold: float = 0.10
    wiggle_window_steps: int = 10
    wiggle_max_reversals: int = 3
    # Equation (13): Martha has no robot-to-robot collision distinction, so a
    # bumper contact and a motor fault share the paper's world penalty. Nearing
    # an obstacle is no longer terminal -- the dense laser penalty handles it.
    goal_reward: float = 100
    collision_penalty: float = 20.0
    timeout_penalty: float = 2.0
    # End policies that stop making meaningful goal progress. At the normal
    # 10 Hz control rate, 200 steps correspond to roughly twenty seconds.
    stagnation_window_steps: int = 200
    stagnation_min_progress: float = 0.10
    stagnation_penalty: float = 2.0


@dataclass(frozen=True)
class RewardState:
    """Minimal history required by the paper's episodic reward terms."""

    previous_distance: float
    best_distance: float
    previous_turn: int = 0
    reversals: tuple[int, ...] = ()

    @classmethod
    def initial(cls, distance: float) -> "RewardState":
        if not math.isfinite(distance) or distance < 0.0:
            raise ValueError("initial reward distance must be finite and nonnegative")
        return cls(previous_distance=float(distance), best_distance=float(distance))


def empty_reward_components() -> dict[str, float]:
    """Return a zero-valued explanation compatible with the training CSV."""
    return {name: 0.0 for name in REWARD_COMPONENT_NAMES}


def validate_reward_config(config: RewardConfig) -> None:
    """Reject invalid paper-reward constants before a run starts."""
    values = (
        config.step_penalty,
        config.distance_positive_scale,
        config.distance_negative_scale,
        config.orientation_positive_scale,
        config.orientation_negative_scale,
        config.shortest_distance_scale,
        config.laser_penalty_scale,
        config.laser_penalty_midpoint,
        config.laser_penalty_steepness,
        config.velocity_penalty_scale,
        config.velocity_slow_distance,
        config.velocity_penalty_steepness,
        config.goal_hold_bonus,
        config.goal_leave_penalty,
        config.wiggle_penalty,
        config.wiggle_turn_rate_threshold,
        config.goal_reward,
        config.collision_penalty,
        config.timeout_penalty,
        config.stagnation_min_progress,
        config.stagnation_penalty,
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("reward scales and penalties must be finite and nonnegative")
    if config.laser_penalty_midpoint <= 0.0:
        raise ValueError("laser_penalty_midpoint must be positive")
    if config.laser_penalty_steepness <= 0.0:
        raise ValueError("laser_penalty_steepness must be positive")
    if config.velocity_slow_distance <= 0.0:
        raise ValueError("velocity_slow_distance must be positive")
    if config.velocity_penalty_steepness <= 0.0:
        raise ValueError("velocity_penalty_steepness must be positive")
    if config.wiggle_window_steps <= 0:
        raise ValueError("wiggle_window_steps must be positive")
    if config.wiggle_max_reversals < 0:
        raise ValueError("wiggle_max_reversals cannot be negative")
    if config.stagnation_window_steps <= 0:
        raise ValueError("stagnation_window_steps must be positive")
    if config.stagnation_min_progress <= 0.0:
        raise ValueError("stagnation_min_progress must be positive")


def update_stagnation(
    reference_distance: float,
    steps_without_progress: int,
    distance: float,
    config: RewardConfig,
) -> tuple[float, int, bool]:
    """Update the meaningful-progress window and report stagnation."""
    validate_reward_config(config)
    if not all(math.isfinite(value) and value >= 0.0 for value in (
        reference_distance,
        distance,
    )):
        raise ValueError("stagnation distances must be finite and nonnegative")
    if steps_without_progress < 0:
        raise ValueError("steps_without_progress cannot be negative")

    progress = reference_distance - distance
    if progress >= config.stagnation_min_progress:
        return float(distance), 0, False
    next_steps = int(steps_without_progress) + 1
    return (
        float(reference_distance),
        next_steps,
        next_steps >= config.stagnation_window_steps,
    )


def _turn_direction(angular_velocity: float, threshold: float) -> int:
    if angular_velocity > threshold:
        return 1
    if angular_velocity < -threshold:
        return -1
    return 0


def _updated_wiggle_state(
    state: RewardState,
    angular_velocity: float,
    config: RewardConfig,
) -> tuple[RewardState, float]:
    """Apply the paper's direct left/right reversal rule in a finite window."""
    turn = _turn_direction(angular_velocity, config.wiggle_turn_rate_threshold)
    reversed_direction = int(turn != 0 and turn == -state.previous_turn)
    reversals = (state.reversals + (reversed_direction,))[
        -config.wiggle_window_steps:
    ]
    wiggle = (
        -config.wiggle_penalty
        if sum(reversals) > config.wiggle_max_reversals
        else 0.0
    )
    # A straight command deliberately breaks the left/right sequence.
    return (
        RewardState(
            previous_distance=state.previous_distance,
            best_distance=state.best_distance,
            previous_turn=turn,
            reversals=reversals,
        ),
        wiggle,
    )


def calculate_reward(
    *,
    state: RewardState,
    distance: float,
    goal_bearing: float,
    minimum_scan: float,
    angular_velocity: float,
    linear_speed: float = 0.0,
    within_goal: bool = False,
    left_goal: bool = False,
    reached_goal: bool,
    collision: bool,
    timeout: bool,
    stagnated: bool = False,
    config: RewardConfig = RewardConfig(),
) -> tuple[float, dict[str, float], RewardState]:
    """
    Calculate one paper reward and return the state for the next step.

    Final states are exclusive, as specified in equation (2): they never mix
    a dense reward with a goal, collision or timeout reward.  Collision keeps
    priority over a simultaneous goal signal to retain Martha's safety rule.
    """
    validate_reward_config(config)
    components = empty_reward_components()
    if collision:
        components["terminal"] = -config.collision_penalty
        return float(components["terminal"]), components, state
    if reached_goal:
        components["terminal"] = config.goal_reward
        return float(components["terminal"]), components, state
    if stagnated:
        components["terminal"] = -config.stagnation_penalty
        return float(components["terminal"]), components, state
    if timeout:
        components["terminal"] = -config.timeout_penalty
        return float(components["terminal"]), components, state

    components["step"] = -config.step_penalty

    if not math.isfinite(distance) or distance < 0.0:
        # A valid terminal state above can still be scored even if an upstream
        # sensor value is invalid.  Non-terminal invalid distances get no
        # geometric reward and preserve their prior state.
        return 0.0, components, state
    if not math.isfinite(goal_bearing):
        goal_bearing = 0.0
    if not math.isfinite(minimum_scan):
        minimum_scan = math.inf
    if not math.isfinite(angular_velocity):
        angular_velocity = 0.0

    delta_distance = state.previous_distance - distance
    components["distance"] = delta_distance * (
        config.distance_negative_scale
        if delta_distance < 0.0
        else config.distance_positive_scale
    )

    angle = min(abs(goal_bearing), math.pi)
    normalized_orientation = 1.0 - 2.0 * angle / math.pi
    components["orientation"] = (
        normalized_orientation * config.orientation_positive_scale
        if normalized_orientation >= 0.0
        else 0.0
    )

    best_distance = state.best_distance
    if distance < best_distance:
        components["shortest_distance"] = (
            (best_distance - distance) * config.shortest_distance_scale
        )
        best_distance = distance

    # Dense sigmoid clearance penalty. The stable two-branch sigmoid keeps a
    # far obstacle (minimum_scan == inf -> argument == -inf) at exactly zero
    # without overflowing math.exp.
    argument = config.laser_penalty_steepness * (
        config.laser_penalty_midpoint - minimum_scan
    )
    if argument >= 0.0:
        saturation = 1.0 / (1.0 + math.exp(-argument))
    else:
        exponential = math.exp(argument)
        saturation = exponential / (1.0 + exponential)
    components["laser"] = -config.laser_penalty_scale * saturation

    # Approach braking: penalize linear speed, gated by a sigmoid so it only
    # bites near the goal. Same stable two-branch sigmoid as the laser term.
    if not math.isfinite(linear_speed):
        linear_speed = 0.0
    gate_argument = config.velocity_penalty_steepness * (
        config.velocity_slow_distance - distance
    )
    if gate_argument >= 0.0:
        gate = 1.0 / (1.0 + math.exp(-gate_argument))
    else:
        gate_exponential = math.exp(gate_argument)
        gate = gate_exponential / (1.0 + gate_exponential)
    components["velocity"] = (
        -config.velocity_penalty_scale * (linear_speed * linear_speed) * gate
    )

    # Optional hold shaping (both scales default to 0.0, so this is inert). A
    # per-step bonus while sitting inside the tolerance, and a one-off penalty
    # on the step the robot leaves it again, so bouncing in/out cannot farm the
    # bonus. Reaching (dwell complete) is terminal above and never lands here.
    if within_goal:
        components["hold"] = config.goal_hold_bonus
    elif left_goal:
        components["hold"] = -config.goal_leave_penalty

    wiggle_state, components["wiggle"] = _updated_wiggle_state(
        state,
        angular_velocity,
        config,
    )
    next_state = RewardState(
        previous_distance=float(distance),
        best_distance=float(best_distance),
        previous_turn=wiggle_state.previous_turn,
        reversals=wiggle_state.reversals,
    )
    reward = float(sum(components.values()))
    if not math.isfinite(reward):
        raise FloatingPointError("reward contains NaN or infinity")
    return reward, components, next_state
