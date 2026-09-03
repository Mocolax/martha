"""One Gazebo process coordinating a namespaced fleet of Martha agents."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Iterable

import numpy as np

from .martha_env import (
    MarthaEnv,
    PendingGazeboReset,
    PendingStep,
    ResetSafetyCheck,
    SensorSnapshot,
    external_contact_models,
)
from .training_layout import (
    MIN_START_SEPARATION,
    WORLD_ORIGINS,
    create_combined_training_world,
    load_training_points,
    parking_pose,
    sample_round_episodes,
    validate_training_points,
)
from .world_map import WorldMap, discover_worlds


SHARED_RESET_SAME_EPISODE_ATTEMPTS = 3
SHARED_RESET_RESAMPLED_ATTEMPTS = 2
SHARED_RESET_MAX_SAMPLES_PER_ATTEMPT = 12
RECYCLE_PLACEMENT_ATTEMPTS = 64


class UnsafeSharedResetError(RuntimeError):
    """A reset produced fresh data but never reached a stable safe pose."""


class RecyclePlacementUnavailable(RuntimeError):
    """No sampled start is currently separated from every active robot."""


def _stop_launch_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=15.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)


class SharedGazeboEnvironments:
    """Own one gzserver and a batch of namespaced Martha environments."""

    def __init__(
        self,
        *,
        count: int,
        sim_speed_factor: float,
        physics_step_size: float,
        lidar_samples: int,
        training_kinematic: bool,
        show_gui: bool,
        startup_timeout: float,
        run_directory: Path,
        worlds_directory: Path,
        points_path: Path,
        environment_kwargs: dict[str, Any],
    ) -> None:
        self._launch_process: subprocess.Popen[Any] | None = None
        self._log_stream: Any | None = None
        self._combined_world: Path | None = None
        self.environments: list[MarthaEnv] = []
        self.count = int(count)
        if self.count <= 0:
            raise ValueError("shared Gazebo requires at least one robot")

        world_paths = discover_worlds(worlds_directory)
        local_maps = tuple(WorldMap.from_sdf(path) for path in world_paths)
        catalog = load_training_points(points_path)
        validate_training_points(
            catalog,
            local_maps,
            robot_count=self.count,
            min_goal_distance=float(environment_kwargs["min_goal_distance"]),
        )
        self.local_maps = local_maps
        self.catalog = catalog
        self._distance_field_caches = [dict() for _ in local_maps]
        self._combined_world = create_combined_training_world(world_paths)

        log_path = run_directory / "gazebo.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_stream = log_path.open("a", encoding="utf-8")
        command = [
            "ros2",
            "launch",
            "martha",
            "simulation.launch.py",
            f"gui:={'true' if show_gui else 'false'}",
            f"sim_speed_factor:={sim_speed_factor}",
            f"physics_step_size:={physics_step_size}",
            f"lidar_samples:={lidar_samples}",
            "lidar_visualize:=false",
            f"training_kinematic:={'true' if training_kinematic else 'false'}",
            f"world:={self._combined_world}",
            f"robot_count:={self.count}",
            "force_namespaced_fleet:=true",
        ]
        try:
            self._launch_process = subprocess.Popen(
                command,
                stdout=self._log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            for index in range(self.count):
                namespace = f"martha_{index}"
                kwargs = dict(environment_kwargs)
                kwargs.setdefault("sensor_timeout", 15.0)
                kwargs.setdefault("control_timeout", 30.0)
                kwargs.setdefault("service_timeout", 15.0)
                kwargs.update(
                    worlds_directory=worlds_directory,
                    world_origins=WORLD_ORIGINS,
                    preloaded_worlds=True,
                    scan_topic=f"/{namespace}/scan",
                    odometry_topic=f"/{namespace}/odometry/filtered",
                    goal_topic=f"/{namespace}/goal_pose",
                    cmd_vel_topic=f"/{namespace}/cmd_vel",
                    contact_topic=f"/{namespace}/contacts",
                    odom_frame=f"{namespace}/odom",
                    robot_name=namespace,
                    service_namespace=namespace,
                )
                self.environments.append(MarthaEnv(**kwargs))
            self._wait_until_ready(startup_timeout, log_path)
            self.park_all()
        # KeyboardInterrupt and SystemExit do not derive from Exception. If
        # either arrives while readiness is still inside this constructor,
        # train_gazebo has not entered its own try/finally yet. Clean up here
        # for every escaping condition so the detached launch session cannot
        # leave a gzserver behind on the default Gazebo master.
        except BaseException:
            self.close()
            raise

    @property
    def reference_env(self) -> MarthaEnv:
        return self.environments[0]

    def _wait_until_ready(self, timeout: float, log_path: Path) -> None:
        deadline = time.monotonic() + timeout
        expected = {f"martha_{index}" for index in range(self.count)}
        expected_topics = {
            topic
            for index in range(self.count)
            for topic in (
                f"/martha_{index}/scan",
                f"/martha_{index}/contacts",
                f"/martha_{index}/odometry/filtered",
            )
        }
        activity_baselines = [
            environment.ros.stream_activity()
            for environment in self.environments
        ]
        startup_unpaused = False
        last_names: set[str] = set()
        last_topics: set[str] = set()
        last_missing_activity: dict[str, tuple[str, ...]] = {}
        while True:
            if self._launch_process is None or self._launch_process.poll() is not None:
                raise RuntimeError(
                    f"shared Gazebo exited during startup; inspect {log_path}"
                )
            try:
                # Make startup explicit instead of relying on gazebo.launch's
                # pause argument. Controller activation is completed by a
                # Gazebo update cycle, so a paused server deadlocks its own
                # spawners.
                if not startup_unpaused:
                    self.reference_env._call_empty("unpause")
                    startup_unpaused = True
                last_names = self.reference_env.ros.gazebo_model_names()
                last_topics = {
                    name
                    for name, _ in self.reference_env.ros.get_topic_names_and_types()
                }
                last_missing_activity = {
                    environment.robot_name: missing
                    for environment, baseline in zip(
                        self.environments,
                        activity_baselines,
                    )
                    if (
                        missing := environment.ros.stream_activity().missing_after(
                            baseline,
                            require_contact=True,
                        )
                    )
                }
                if (
                    expected <= last_names
                    and expected_topics <= last_topics
                    and not last_missing_activity
                ):
                    self.reference_env._call_empty("pause")
                    return
            except (RuntimeError, TimeoutError):
                pass
            if time.monotonic() >= deadline:
                problems = []
                missing_models = sorted(expected - last_names)
                missing_topics = sorted(expected_topics - last_topics)
                if missing_models:
                    problems.append(f"models={missing_models}")
                if missing_topics:
                    problems.append(f"topics={missing_topics}")
                problems.extend(
                    f"{robot_name}=[{', '.join(missing)}]"
                    for robot_name, missing in sorted(last_missing_activity.items())
                )
                raise TimeoutError(
                    "shared Gazebo did not produce startup activity: "
                    f"{'; '.join(problems) or 'unknown readiness failure'}; "
                    f"inspect {log_path}"
                )
            time.sleep(0.25)

    def park(self, index: int) -> None:
        """Move one inactive robot to its unique external parking position."""
        x, y, yaw = parking_pose(index, self.count)
        self.environments[index].park(x, y, yaw)

    def park_all(self, except_indices: Iterable[int] = ()) -> None:
        """Park every robot not explicitly kept active."""
        excluded = set(except_indices)
        self.reference_env._call_empty("pause")
        for index in range(self.count):
            if index not in excluded:
                self.park(index)

    def _sample_reset_episodes(
        self,
        *,
        world_index: int,
        seeds: list[int],
        resample_index: int,
        max_goal_distance: float | None = None,
    ) -> tuple[Any, ...]:
        """Sample one deterministic round, preserving the original first draw."""
        local_map = self.local_maps[world_index]
        seed_entropy = (
            seeds if resample_index == 0 else [*seeds, int(resample_index)]
        )
        round_rng = np.random.default_rng(np.random.SeedSequence(seed_entropy))
        return sample_round_episodes(
            local_map,
            self.catalog[local_map.world_name],
            WORLD_ORIGINS[local_map.world_name],
            robot_count=len(seeds),
            min_goal_distance=self.reference_env.min_goal_distance,
            max_goal_distance=max_goal_distance,
            rng=round_rng,
            distance_field_cache=self._distance_field_caches[world_index],
        )

    def _sample_recycled_episodes(
        self,
        *,
        world_index: int,
        seeds: list[int],
        active_indices: Iterable[int],
        max_goal_distance: float | None = None,
    ) -> tuple[Any, ...]:
        """Sample deterministic starts separated from every active robot."""
        occupied = []
        for index in active_indices:
            position = self.environments[index].ground_truth_position()
            if position is None:
                raise RuntimeError(
                    "cannot recycle a slot without active ground-truth poses"
                )
            occupied.append(position)

        for resample_index in range(RECYCLE_PLACEMENT_ATTEMPTS):
            episodes = self._sample_reset_episodes(
                world_index=world_index,
                seeds=seeds,
                resample_index=resample_index,
                max_goal_distance=max_goal_distance,
            )
            if all(
                np.hypot(episode.start_x - x, episode.start_y - y)
                >= MIN_START_SEPARATION
                for episode in episodes
                for x, y in occupied
            ):
                return episodes
        raise RecyclePlacementUnavailable(
            "could not recycle robots away from every active robot after "
            f"{RECYCLE_PLACEMENT_ATTEMPTS} deterministic samples"
        )

    def _contact_collision_indices(
        self,
        pending: dict[int, PendingStep],
    ) -> set[int]:
        """Return every active fleet index involved in a fresh contact."""
        collisions: set[int] = set()
        robot_indices = {
            environment.robot_name: index
            for index, environment in enumerate(self.environments)
        }
        for index, step in pending.items():
            environment = self.environments[index]
            external = external_contact_models(
                environment.ros.consume_contact_pairs(
                    after_sequence=step.contact_sequence,
                ),
                environment.robot_name,
            )
            if external:
                collisions.add(index)
            collisions.update(
                robot_indices[name] for name in external if name in robot_indices
            )
        return collisions

    def reset_round(
        self,
        *,
        world_index: int,
        seeds: list[int],
        max_goal_distance: float | None = None,
    ) -> dict[int, tuple[np.ndarray, dict[str, Any]]]:
        """Start one group round in a shared map from catalog points."""
        active_count = len(seeds)
        if not 0 < active_count <= self.count:
            raise ValueError("round seed count must fit the available robots")
        total_attempts = (
            SHARED_RESET_SAME_EPISODE_ATTEMPTS
            + SHARED_RESET_RESAMPLED_ATTEMPTS
        )
        base_episodes = self._sample_reset_episodes(
            world_index=world_index,
            seeds=seeds,
            resample_index=0,
            max_goal_distance=max_goal_distance,
        )
        failures: list[str] = []

        for attempt_index in range(total_attempts):
            resample_index = max(
                0,
                attempt_index - SHARED_RESET_SAME_EPISODE_ATTEMPTS + 1,
            )
            episodes = (
                base_episodes
                if resample_index == 0
                else self._sample_reset_episodes(
                    world_index=world_index,
                    seeds=seeds,
                    resample_index=resample_index,
                    max_goal_distance=max_goal_distance,
                )
            )
            self.park_all()
            # A failed attempt never commits only part of a fleet round.
            self.reference_env._reset_world_scenario(world_index)
            pending: dict[int, PendingGazeboReset] = {}
            for index, (seed, episode) in enumerate(zip(seeds, episodes)):
                options = {
                    "world_index": world_index,
                    "start": (
                        episode.start_x,
                        episode.start_y,
                        episode.start_yaw,
                    ),
                    "goal": (episode.goal_x, episode.goal_y),
                }
                pending[index] = self.environments[index].prepare_shared_reset(
                    seed=seed,
                    options=options,
                )
            self.reference_env._call_empty("unpause")
            try:
                snapshots, validations = self._wait_for_reset_round(pending)
            except (TimeoutError, UnsafeSharedResetError) as exc:
                problem = (
                    f"attempt={attempt_index + 1}/{total_attempts}, "
                    f"resample={resample_index}: {exc}"
                )
                failures.append(problem)
                print(f"WARNING: shared Gazebo reset retry: {problem}", flush=True)
                continue
            finally:
                self.reference_env._call_empty("pause")

            return {
                index: self.environments[index].finish_shared_reset(
                    pending[index],
                    snapshots[index],
                    validation=validations[index],
                )
                for index in pending
            }

        local_map = self.local_maps[world_index]
        raise RuntimeError(
            "shared Gazebo reset exhausted recovery attempts for "
            f"map={local_map.world_name}: "
            + " | ".join(failures)
        )

    def reset_mixed_round(
        self,
        *,
        world_indices: list[int],
        seeds: list[int],
        max_goal_distance: float | None = None,
    ) -> dict[int, tuple[np.ndarray, dict[str, Any]]]:
        """Start one fleet round with every robot on a distinct map island."""
        active_count = len(seeds)
        if len(world_indices) != active_count:
            raise ValueError("world_indices and seeds must have equal length")
        if not 0 < active_count <= self.count:
            raise ValueError("round seed count must fit the available robots")
        if len(set(world_indices)) != len(world_indices):
            raise ValueError("mixed training rounds require distinct maps")
        if any(index < 0 or index >= len(self.local_maps) for index in world_indices):
            raise IndexError("mixed round world index is outside the map catalog")

        total_attempts = (
            SHARED_RESET_SAME_EPISODE_ATTEMPTS
            + SHARED_RESET_RESAMPLED_ATTEMPTS
        )
        failures: list[str] = []
        base_episodes = tuple(
            self._sample_reset_episodes(
                world_index=world_index,
                seeds=[seed],
                resample_index=0,
                max_goal_distance=max_goal_distance,
            )[0]
            for world_index, seed in zip(world_indices, seeds)
        )
        for attempt_index in range(total_attempts):
            resample_index = max(
                0,
                attempt_index - SHARED_RESET_SAME_EPISODE_ATTEMPTS + 1,
            )
            episodes = (
                base_episodes
                if resample_index == 0
                else tuple(
                    self._sample_reset_episodes(
                        world_index=world_index,
                        seeds=[seed],
                        resample_index=resample_index,
                        max_goal_distance=max_goal_distance,
                    )[0]
                    for world_index, seed in zip(world_indices, seeds)
                )
            )
            self.park_all()
            pending: dict[int, PendingGazeboReset] = {}
            for index, (world_index, seed, episode) in enumerate(
                zip(world_indices, seeds, episodes)
            ):
                pending[index] = self.environments[index].prepare_shared_reset(
                    seed=seed,
                    options={
                        "world_index": world_index,
                        "start": (
                            episode.start_x,
                            episode.start_y,
                            episode.start_yaw,
                        ),
                        "goal": (episode.goal_x, episode.goal_y),
                    },
                )
            self.reference_env._call_empty("unpause")
            try:
                snapshots, validations = self._wait_for_reset_round(pending)
            except (TimeoutError, UnsafeSharedResetError) as exc:
                problem = (
                    f"attempt={attempt_index + 1}/{total_attempts}, "
                    f"resample={resample_index}: {exc}"
                )
                failures.append(problem)
                print(
                    f"WARNING: mixed Gazebo reset retry: {problem}",
                    flush=True,
                )
                continue
            finally:
                self.reference_env._call_empty("pause")
            return {
                index: self.environments[index].finish_shared_reset(
                    pending[index],
                    snapshots[index],
                    validation=validations[index],
                )
                for index in pending
            }
        names = [self.local_maps[index].world_name for index in world_indices]
        raise RuntimeError(
            "mixed Gazebo reset exhausted recovery attempts for maps="
            f"{names}: "
            + " | ".join(failures)
        )

    def reset_slots(
        self,
        *,
        world_index: int,
        assignments: dict[int, int],
        active_indices: Iterable[int],
        max_goal_distance: float | None = None,
    ) -> tuple[
        dict[int, tuple[np.ndarray, dict[str, Any]]],
        dict[int, tuple[np.ndarray, float, bool, bool, dict[str, Any]]],
    ]:
        """Recycle inactive slots while accounting active robots' stop step."""
        slots = sorted(assignments)
        active = sorted(set(active_indices))
        if not slots:
            return {}, {}
        if set(slots) & set(active):
            raise ValueError("recycled and active robot indices must be disjoint")
        if any(index < 0 or index >= self.count for index in slots + active):
            raise ValueError("fleet index is outside the configured robot count")

        seeds = [assignments[index] for index in slots]
        episodes = self._sample_recycled_episodes(
            world_index=world_index,
            seeds=seeds,
            active_indices=active,
            max_goal_distance=max_goal_distance,
        )
        pending_resets: dict[int, PendingGazeboReset] = {}
        for index, seed, episode in zip(slots, seeds, episodes):
            self.park(index)
            pending_resets[index] = self.environments[index].prepare_shared_reset(
                seed=seed,
                options={
                    "world_index": world_index,
                    "start": (
                        episode.start_x,
                        episode.start_y,
                        episode.start_yaw,
                    ),
                    "goal": (episode.goal_x, episode.goal_y),
                },
            )
        pending_stops = {
            index: self.environments[index].prepare_stop_step()
            for index in active
        }

        self.reference_env._call_empty("unpause")
        try:
            reset_snapshots, validations = self._wait_for_reset_round(
                pending_resets
            )
            active_snapshots = {
                index: self.environments[index].wait_for_step_snapshot(step)
                for index, step in pending_stops.items()
            }
        finally:
            self.reference_env._call_empty("pause")

        collisions = self._contact_collision_indices(pending_stops)
        passive_results = {
            index: self.environments[index].finish_step(
                pending_stops[index],
                active_snapshots[index],
                contact_collision=index in collisions,
            )
            for index in active
        }
        for index, transition in passive_results.items():
            if transition[2] or transition[3]:
                self.park(index)

        reset_results = {
            index: self.environments[index].finish_shared_reset(
                pending_resets[index],
                reset_snapshots[index],
                validation=validations[index],
            )
            for index in slots
        }
        return reset_results, passive_results

    def _wait_for_reset_round(
        self,
        pending: dict[int, PendingGazeboReset],
    ) -> tuple[dict[int, SensorSnapshot], dict[int, ResetSafetyCheck]]:
        """
        Collect consecutive safe samples and fresh contact activity.

        ROS callbacks run in each environment's executor. Polling every robot
        non-blockingly avoids both competing pause/unpause calls and an
        accidental ``robot_count * control_timeout`` serial wait.
        """
        if not pending:
            return {}, {}
        deadline = time.monotonic() + max(
            self.environments[index].control_timeout for index in pending
        )
        sequences = {
            index: reset.sequences for index, reset in pending.items()
        }
        observed_counts = {index: 0 for index in pending}
        safe_streaks = {index: 0 for index in pending}
        snapshots: dict[int, SensorSnapshot] = {}
        validations: dict[int, ResetSafetyCheck] = {}
        last_checks: dict[int, ResetSafetyCheck] = {}
        distance_fields = {
            index: reset.world_map.distance_field(
                reset.sample.goal_x,
                reset.sample.goal_y,
            )
            for index, reset in pending.items()
        }
        contact_fresh = {index: False for index in pending}

        while True:
            for index, reset in pending.items():
                environment = self.environments[index]
                required_samples = environment.reset_settle_samples
                if safe_streaks[index] < required_samples:
                    snapshot = environment.ros.wait_for_fresh_snapshot(
                        sequences[index],
                        0.0,
                        use_ground_truth=True,
                        after_stamp_ns=reset.reset_stamp_ns,
                    )
                    if snapshot is not None:
                        observed_counts[index] += 1
                        sequences[index] = (
                            snapshot.scan_sequence,
                            snapshot.odometry_sequence,
                            snapshot.ground_truth_sequence,
                        )
                        check = environment.validate_shared_reset(
                            reset.world_map,
                            reset.sample,
                            snapshot,
                            distance_field=distance_fields[index],
                        )
                        last_checks[index] = check
                        if check.safe:
                            safe_streaks[index] += 1
                            if safe_streaks[index] == required_samples:
                                snapshots[index] = snapshot
                                validations[index] = check
                        else:
                            safe_streaks[index] = 0
                if not contact_fresh[index]:
                    contact_fresh[index] = (
                        environment.ros.stream_activity().contact_sequence
                        > reset.contact_sequence
                    )

            exhausted = [
                index
                for index in pending
                if (
                    safe_streaks[index]
                    < self.environments[index].reset_settle_samples
                    and observed_counts[index]
                    >= SHARED_RESET_MAX_SAMPLES_PER_ATTEMPT
                )
            ]
            if exhausted:
                problems = [
                    self._unsafe_reset_problem(
                        index=index,
                        pending=pending[index],
                        observed=observed_counts[index],
                        safe_streak=safe_streaks[index],
                        check=last_checks.get(index),
                    )
                    for index in exhausted
                ]
                raise UnsafeSharedResetError(
                    "unstable or unsafe reset observations: " + "; ".join(problems)
                )

            incomplete = [
                index
                for index in pending
                if (
                    safe_streaks[index]
                    < self.environments[index].reset_settle_samples
                    or not contact_fresh[index]
                )
            ]
            if not incomplete:
                return snapshots, validations
            if time.monotonic() >= deadline:
                problems = []
                for index in incomplete:
                    reset = pending[index]
                    environment = self.environments[index]
                    missing = []
                    if safe_streaks[index] < environment.reset_settle_samples:
                        missing.extend(
                            environment.ros.missing_fresh_snapshot_data(
                                sequences[index],
                                use_ground_truth=True,
                                after_stamp_ns=reset.reset_stamp_ns,
                            )
                        )
                        if not missing:
                            missing.append(
                                "safe_samples="
                                f"{safe_streaks[index]}/"
                                f"{environment.reset_settle_samples}, "
                                f"observed={observed_counts[index]}/"
                                f"{SHARED_RESET_MAX_SAMPLES_PER_ATTEMPT}"
                            )
                            check = last_checks.get(index)
                            if check is not None:
                                missing.append(check.diagnostic())
                    if not contact_fresh[index]:
                        activity = environment.ros.stream_activity()
                        missing.append(
                            "contact(sequence="
                            f"{activity.contact_sequence}, "
                            f"need>{reset.contact_sequence})"
                        )
                    problems.append(
                        f"{environment.robot_name}=[{', '.join(missing)}]"
                    )
                raise TimeoutError(
                    "shared Gazebo batch reset timed out: "
                    + "; ".join(problems)
                )
            time.sleep(0.01)

    def _unsafe_reset_problem(
        self,
        *,
        index: int,
        pending: PendingGazeboReset,
        observed: int,
        safe_streak: int,
        check: ResetSafetyCheck | None,
    ) -> str:
        """Describe one robot that exhausted its reset observations."""
        environment = self.environments[index]
        sample = pending.sample
        fields = [
            f"map={pending.world_map.world_name}",
            (
                "requested_start=("
                f"{sample.start_x:.6f},{sample.start_y:.6f},"
                f"{sample.start_yaw:.6f})"
            ),
            f"goal=({sample.goal_x:.6f},{sample.goal_y:.6f})",
            (
                f"safe_samples={safe_streak}/"
                f"{environment.reset_settle_samples}"
            ),
            f"observed={observed}/{SHARED_RESET_MAX_SAMPLES_PER_ATTEMPT}",
        ]
        if check is not None:
            fields.append(check.diagnostic())
        return f"{environment.robot_name}=[{', '.join(fields)}]"

    def reset_single(
        self,
        *,
        robot_index: int,
        world_index: int,
        seed: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset one evaluation robot while every other robot stays parked."""
        self.park_all()
        local_map = self.local_maps[world_index]
        episode = sample_round_episodes(
            local_map,
            self.catalog[local_map.world_name],
            WORLD_ORIGINS[local_map.world_name],
            robot_count=1,
            min_goal_distance=self.reference_env.min_goal_distance,
            rng=np.random.default_rng(seed),
            distance_field_cache=self._distance_field_caches[world_index],
        )[0]
        result = self.environments[robot_index].reset(
            seed=seed,
            options={
                "world_index": world_index,
                "start": (episode.start_x, episode.start_y, episode.start_yaw),
                "goal": (episode.goal_x, episode.goal_y),
            },
        )
        self.environments[robot_index].ros.clear_contacts()
        return result

    def step_batch(
        self,
        actions: dict[int, np.ndarray],
    ) -> dict[int, tuple[np.ndarray, float, bool, bool, dict[str, Any]]]:
        """Advance all active robots through one shared physics interval."""
        if not actions:
            return {}
        pending = {
            index: self.environments[index].prepare_step(action)
            for index, action in actions.items()
        }
        self.reference_env._call_empty("unpause")
        try:
            snapshots = {
                index: self.environments[index].wait_for_step_snapshot(step)
                for index, step in pending.items()
            }
        finally:
            self.reference_env._call_empty("pause")

        collisions = self._contact_collision_indices(pending)

        results = {
            index: self.environments[index].finish_step(
                pending[index],
                snapshots[index],
                contact_collision=index in collisions,
            )
            for index in pending
        }
        for index, transition in results.items():
            if transition[2] or transition[3]:
                self.park(index)
        return results

    def close(self) -> None:
        """Release every ROS environment, launch process and temporary world."""
        for environment in self.environments:
            try:
                environment.close()
            except Exception:
                pass
        self.environments.clear()
        _stop_launch_process(self._launch_process)
        self._launch_process = None
        if self._log_stream is not None:
            self._log_stream.close()
            self._log_stream = None
        if self._combined_world is not None:
            self._combined_world.unlink(missing_ok=True)
            self._combined_world = None
