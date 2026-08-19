"""One Gazebo process coordinating a namespaced fleet of Martha agents."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Iterable

import numpy as np

from .martha_env import MarthaEnv, external_contact_models
from .training_layout import (
    WORLD_ORIGINS,
    create_combined_training_world,
    load_training_points,
    parking_pose,
    sample_round_episodes,
    validate_training_points,
)
from .world_map import WorldMap, discover_worlds


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
            f"world:={self._combined_world}",
            f"robot_count:={self.count}",
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
        except Exception:
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
                f"/martha_{index}/mecanum_drive_controller/odometry",
                f"/martha_{index}/joint_states",
            )
        }
        startup_unpaused = False
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
                names = self.reference_env.ros.gazebo_model_names()
                live_topics = {
                    name
                    for name, _ in self.reference_env.ros.get_topic_names_and_types()
                }
                if expected <= names and expected_topics <= live_topics:
                    self.reference_env._call_empty("pause")
                    return
            except (RuntimeError, TimeoutError):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"shared Gazebo did not become ready; inspect {log_path}"
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

    def reset_round(
        self,
        *,
        world_index: int,
        seeds: list[int],
    ) -> dict[int, tuple[np.ndarray, dict[str, Any]]]:
        """Start one group round in a shared map from catalog points."""
        active_count = len(seeds)
        if not 0 < active_count <= self.count:
            raise ValueError("round seed count must fit the available robots")
        local_map = self.local_maps[world_index]
        round_rng = np.random.default_rng(np.random.SeedSequence(seeds))
        episodes = sample_round_episodes(
            local_map,
            self.catalog[local_map.world_name],
            WORLD_ORIGINS[local_map.world_name],
            robot_count=active_count,
            min_goal_distance=self.reference_env.min_goal_distance,
            rng=round_rng,
            distance_field_cache=self._distance_field_caches[world_index],
        )
        self.park_all()
        results = {}
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
            results[index] = self.environments[index].reset(
                seed=seed,
                options=options,
            )
            self.environments[index].ros.clear_contacts()
        self.reference_env._call_empty("pause")
        return results

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

        collisions: set[int] = set()
        robot_indices = {
            environment.robot_name: index
            for index, environment in enumerate(self.environments)
        }
        for index in pending:
            environment = self.environments[index]
            external = external_contact_models(
                environment.ros.consume_contact_pairs(
                    after_sequence=pending[index].contact_sequence,
                ),
                environment.robot_name,
            )
            if external:
                collisions.add(index)
            collisions.update(
                robot_indices[name] for name in external if name in robot_indices
            )

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
