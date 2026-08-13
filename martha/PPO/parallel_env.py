"""Isolated multi-process Gazebo environments for vectorized PPO rollout."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
from typing import Any

import numpy as np
from gymnasium import spaces

from .actions import ActionLimits


@dataclass(frozen=True)
class ParallelWorkerConfig:
    """Serializable launch and MarthaEnv settings for one worker."""

    index: int
    ros_domain_id: int
    gazebo_port: int
    sim_speed_factor: float
    gui: bool
    startup_timeout: float
    log_path: str
    environment_kwargs: dict[str, Any]


class RemoteEnvironmentError(RuntimeError):
    """Report a worker-side exception with its original traceback."""


def _simulation_launch_command(config: ParallelWorkerConfig) -> list[str]:
    """Build the isolated simulation command for one rollout worker."""
    return [
        "ros2",
        "launch",
        "martha",
        "simulation.launch.py",
        f"gui:={'true' if config.gui else 'false'}",
        f"sim_speed_factor:={config.sim_speed_factor}",
    ]


def _stop_launch_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=10.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)


def _worker_main(connection: Any, config: ParallelWorkerConfig) -> None:
    """Own one ROS domain, Gazebo server and MarthaEnv behind a pipe."""
    def graceful_termination(_signum: int, _frame: Any) -> None:
        raise SystemExit

    signal.signal(signal.SIGTERM, graceful_termination)
    os.environ["ROS_DOMAIN_ID"] = str(config.ros_domain_id)
    os.environ["GAZEBO_MASTER_URI"] = (
        f"http://127.0.0.1:{config.gazebo_port}"
    )
    os.environ["RCUTILS_LOGGING_BUFFERED_STREAM"] = "1"
    launch_process = None
    environment = None
    log_stream = None
    try:
        log_path = Path(config.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_stream = log_path.open("a", encoding="utf-8")
        launch_process = subprocess.Popen(
            _simulation_launch_command(config),
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            start_new_session=True,
        )

        # Import after setting the domain so rclpy initializes in this worker's
        # isolated DDS graph.
        from .martha_env import MarthaEnv

        environment = MarthaEnv(**config.environment_kwargs)
        deadline = time.monotonic() + config.startup_timeout
        while True:
            if launch_process.poll() is not None:
                raise RuntimeError(
                    f"Gazebo worker {config.index} exited during startup; "
                    f"inspect {config.log_path}"
                )
            try:
                environment._call_empty("pause")
                break
            except (RuntimeError, TimeoutError):
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Gazebo worker {config.index} did not become ready; "
                        f"inspect {config.log_path}"
                    )
                time.sleep(0.25)

        metadata = {
            "observation_low": environment.observation_space.low,
            "observation_high": environment.observation_space.high,
            "action_low": environment.action_space.low,
            "action_high": environment.action_space.high,
            "action_limits": asdict(environment.action_limits),
            "policy_contract": environment.policy_contract,
            "max_goal_distance": environment.max_goal_distance,
            "world_count": len(environment.predefined_maps),
        }
        connection.send(("ready", metadata))

        while True:
            request = connection.recv()
            command = request[0]
            try:
                if command == "reset":
                    result = environment.reset(
                        seed=request[1],
                        options=request[2],
                    )
                elif command == "step":
                    result = environment.step(request[1])
                elif command == "stop":
                    environment.stop()
                    result = None
                elif command == "close":
                    connection.send(("ok", None))
                    break
                else:
                    raise ValueError(f"unknown worker command: {command}")
                connection.send(("ok", result))
            except Exception as exc:
                connection.send(
                    (
                        "error",
                        f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                    )
                )
    except EOFError:
        pass
    except BaseException as exc:
        try:
            connection.send(
                (
                    "startup_error",
                    f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                )
            )
        except Exception:
            pass
    finally:
        if environment is not None:
            try:
                environment.close()
            except Exception:
                pass
        _stop_launch_process(launch_process)
        if log_stream is not None:
            log_stream.close()
        connection.close()


class RemoteMarthaEnv:
    """Parent-side proxy exposing the MarthaEnv interface used by training."""

    def __init__(self, connection: Any, process: Any, metadata: dict[str, Any]):
        self._connection = connection
        self._process = process
        self._pending = False
        self.observation_space = spaces.Box(
            low=np.asarray(metadata["observation_low"], dtype=np.float32),
            high=np.asarray(metadata["observation_high"], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.asarray(metadata["action_low"], dtype=np.float32),
            high=np.asarray(metadata["action_high"], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_limits = ActionLimits(**metadata["action_limits"])
        self.policy_contract = dict(metadata["policy_contract"])
        self.max_goal_distance = float(metadata["max_goal_distance"])
        self.predefined_maps = tuple(range(int(metadata["world_count"])))

    def _send(self, request: tuple[Any, ...]) -> None:
        if self._pending:
            raise RuntimeError("remote environment already has a pending request")
        if not self._process.is_alive():
            raise RemoteEnvironmentError("Gazebo worker process is not alive")
        self._connection.send(request)
        self._pending = True

    def _receive(self) -> Any:
        if not self._pending:
            raise RuntimeError("remote environment has no pending request")
        try:
            status, payload = self._connection.recv()
        except EOFError as exc:
            raise RemoteEnvironmentError("Gazebo worker disconnected") from exc
        finally:
            self._pending = False
        if status != "ok":
            raise RemoteEnvironmentError(payload)
        return payload

    def reset(self, *, seed: int, options: dict[str, Any]):
        self.send_reset(seed=seed, options=options)
        return self.receive_reset()

    def send_reset(self, *, seed: int, options: dict[str, Any]) -> None:
        self._send(("reset", seed, options))

    def receive_reset(self):
        return self._receive()

    def send_step(self, action: np.ndarray) -> None:
        self._send(("step", np.asarray(action, dtype=np.float32)))

    def receive_step(self):
        return self._receive()

    def step(self, action: np.ndarray):
        self.send_step(action)
        return self.receive_step()

    def stop(self) -> None:
        self._send(("stop",))
        self._receive()

    def close(self) -> None:
        if self._process.is_alive():
            try:
                self._send(("close",))
                self._receive()
            except Exception:
                pass


class ParallelGazeboEnvironments:
    """Launch and own a configurable collection of isolated Gazebo workers."""

    def __init__(
        self,
        *,
        count: int,
        ros_domain_base: int,
        gazebo_port_base: int,
        sim_speed_factor: float,
        show_gui: bool,
        startup_timeout: float,
        run_directory: Path,
        environment_kwargs: dict[str, Any],
    ) -> None:
        context = mp.get_context("spawn")
        self._processes = []
        self.environments: list[RemoteMarthaEnv] = []
        pending = []
        try:
            for index in range(count):
                parent_connection, child_connection = context.Pipe()
                config = ParallelWorkerConfig(
                    index=index,
                    ros_domain_id=ros_domain_base + index,
                    gazebo_port=gazebo_port_base + index,
                    sim_speed_factor=sim_speed_factor,
                    gui=bool(show_gui and index == 0),
                    startup_timeout=startup_timeout,
                    log_path=str(
                        run_directory
                        / "parallel_logs"
                        / f"gazebo_worker_{index}.log"
                    ),
                    environment_kwargs=dict(environment_kwargs),
                )
                process = context.Process(
                    target=_worker_main,
                    args=(child_connection, config),
                    name=f"martha-gazebo-{index}",
                )
                process.start()
                child_connection.close()
                self._processes.append(process)
                pending.append((parent_connection, process, config))

            deadline = time.monotonic() + startup_timeout
            for connection, process, config in pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0 or not connection.poll(remaining):
                    raise TimeoutError(
                        f"Gazebo worker {config.index} startup timed out; "
                        f"inspect {config.log_path}"
                    )
                status, payload = connection.recv()
                if status != "ready":
                    raise RemoteEnvironmentError(payload)
                self.environments.append(
                    RemoteMarthaEnv(connection, process, payload)
                )
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        for environment in self.environments:
            environment.close()
        for process in self._processes:
            process.join(timeout=15.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        self.environments.clear()
        self._processes.clear()
