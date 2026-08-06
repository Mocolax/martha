"""
Geometry helpers for the Gazebo worlds used by PPO.

This module deliberately has no ROS or Gymnasium dependency.  It parses the
simple SDF arenas stored in ``worlds/`` and builds an inflated 2-D free-space
grid.  The same grid is used to sample valid start/goal pairs and to compute a
geodesic potential for reward shaping.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
import math
from pathlib import Path
import re
from typing import Iterable
import xml.etree.ElementTree as ET

import numpy as np


BOUNDARY_NAMES = {"wall_back", "wall_front", "wall_left", "wall_right"}


def _pose_2d(text: str | None) -> tuple[float, float, float]:
    values = [float(value) for value in (text or "0 0 0 0 0 0").split()]
    values.extend([0.0] * (6 - len(values)))
    return values[0], values[1], values[5]


def _compose_pose(
    parent: tuple[float, float, float],
    child: tuple[float, float, float],
) -> tuple[float, float, float]:
    px, py, pyaw = parent
    cx, cy, cyaw = child
    cosine = math.cos(pyaw)
    sine = math.sin(pyaw)
    return (
        px + cosine * cx - sine * cy,
        py + sine * cx + cosine * cy,
        pyaw + cyaw,
    )


@dataclass(frozen=True)
class BoxObstacle:
    """A rectangular collision projected onto the world XY plane."""

    name: str
    x: float
    y: float
    size_x: float
    size_y: float
    yaw: float = 0.0

    @property
    def projected_half_x(self) -> float:
        cosine = abs(math.cos(self.yaw))
        sine = abs(math.sin(self.yaw))
        return 0.5 * (cosine * self.size_x + sine * self.size_y)

    @property
    def projected_half_y(self) -> float:
        cosine = abs(math.cos(self.yaw))
        sine = abs(math.sin(self.yaw))
        return 0.5 * (sine * self.size_x + cosine * self.size_y)

    def contains(self, x: np.ndarray, y: np.ndarray, margin: float) -> np.ndarray:
        """Return a mask for points inside this box inflated by ``margin``."""
        dx = x - self.x
        dy = y - self.y
        cosine = math.cos(self.yaw)
        sine = math.sin(self.yaw)
        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy
        return (
            (np.abs(local_x) <= self.size_x * 0.5 + margin)
            & (np.abs(local_y) <= self.size_y * 0.5 + margin)
        )


@dataclass(frozen=True)
class EpisodeSample:
    start_x: float
    start_y: float
    start_yaw: float
    goal_x: float
    goal_y: float
    shortest_path: float


class WorldMap:
    """Inflated occupancy representation of one Martha SDF scenario."""

    def __init__(
        self,
        path: Path,
        world_name: str,
        bounds: tuple[float, float, float, float],
        obstacles: Iterable[BoxObstacle],
        model_xml: dict[str, str],
        resolution: float = 0.10,
        robot_clearance: float = 0.45,
    ):
        self.path = Path(path)
        self.world_name = world_name
        self.bounds = tuple(float(value) for value in bounds)
        self.obstacles = tuple(obstacles)
        self.model_xml = dict(model_xml)
        self.resolution = float(resolution)
        self.robot_clearance = float(robot_clearance)
        if self.resolution <= 0.0:
            raise ValueError("resolution must be positive")
        if self.robot_clearance < 0.0:
            raise ValueError("robot_clearance cannot be negative")

        min_x, max_x, min_y, max_y = self.bounds
        if not min_x < max_x or not min_y < max_y:
            raise ValueError(f"invalid arena bounds: {self.bounds}")

        self.x_coordinates = np.arange(
            min_x + self.robot_clearance,
            max_x - self.robot_clearance + self.resolution * 0.5,
            self.resolution,
            dtype=np.float32,
        )
        self.y_coordinates = np.arange(
            min_y + self.robot_clearance,
            max_y - self.robot_clearance + self.resolution * 0.5,
            self.resolution,
            dtype=np.float32,
        )
        if len(self.x_coordinates) == 0 or len(self.y_coordinates) == 0:
            raise ValueError("arena is too small for the requested clearance")

        grid_x, grid_y = np.meshgrid(
            self.x_coordinates,
            self.y_coordinates,
            indexing="xy",
        )
        occupied = np.zeros(grid_x.shape, dtype=bool)
        for obstacle in self.obstacles:
            if obstacle.name in BOUNDARY_NAMES:
                continue
            occupied |= obstacle.contains(
                grid_x,
                grid_y,
                margin=self.robot_clearance,
            )
        self.free = ~occupied
        self.components = self._label_components()
        labels, counts = np.unique(self.components[self.free], return_counts=True)
        valid = labels >= 0
        if not np.any(valid):
            raise ValueError(f"scenario {self.path.name} has no free cells")
        labels = labels[valid]
        counts = counts[valid]
        self.component_sizes = {
            int(label): int(count) for label, count in zip(labels, counts)
        }

    @classmethod
    def from_sdf(
        cls,
        path: str | Path,
        resolution: float = 0.10,
        robot_clearance: float = 0.45,
    ) -> "WorldMap":
        path = Path(path).resolve()
        root = ET.parse(path).getroot()
        world = root.find("world")
        if world is None:
            raise ValueError(f"{path} does not contain an SDF world")

        obstacles: list[BoxObstacle] = []
        model_xml: dict[str, str] = {}
        for model in world.findall("model"):
            model_name = model.get("name", "")
            if not model_name:
                continue
            model_xml[model_name] = ET.tostring(model, encoding="unicode")
            model_pose = _pose_2d(model.findtext("pose"))
            for link_index, link in enumerate(model.findall("link")):
                link_pose = _compose_pose(
                    model_pose,
                    _pose_2d(link.findtext("pose")),
                )
                for collision_index, collision in enumerate(link.findall("collision")):
                    size_element = collision.find("geometry/box/size")
                    if size_element is None or not size_element.text:
                        continue
                    sizes = [float(value) for value in size_element.text.split()]
                    if len(sizes) < 2:
                        continue
                    collision_pose = _compose_pose(
                        link_pose,
                        _pose_2d(collision.findtext("pose")),
                    )
                    suffix = "" if link_index == collision_index == 0 else (
                        f"_{link_index}_{collision_index}"
                    )
                    obstacles.append(
                        BoxObstacle(
                            name=model_name + suffix,
                            x=collision_pose[0],
                            y=collision_pose[1],
                            size_x=sizes[0],
                            size_y=sizes[1],
                            yaw=collision_pose[2],
                        )
                    )

        by_name = {obstacle.name: obstacle for obstacle in obstacles}
        missing = BOUNDARY_NAMES - by_name.keys()
        if missing:
            raise ValueError(
                f"{path.name} is missing arena boundaries: {sorted(missing)}"
            )
        bounds = (
            by_name["wall_back"].x + by_name["wall_back"].projected_half_x,
            by_name["wall_front"].x - by_name["wall_front"].projected_half_x,
            by_name["wall_right"].y + by_name["wall_right"].projected_half_y,
            by_name["wall_left"].y - by_name["wall_left"].projected_half_y,
        )
        return cls(
            path=path,
            world_name=world.get("name", path.stem),
            bounds=bounds,
            obstacles=obstacles,
            model_xml=model_xml,
            resolution=resolution,
            robot_clearance=robot_clearance,
        )

    @property
    def scenario_model_names(self) -> tuple[str, ...]:
        return tuple(name for name in self.model_xml if name != "goal_point")

    def wrapped_model_sdf(self, model_name: str) -> str:
        return (
            '<?xml version="1.0"?>'
            '<sdf version="1.6">'
            f'{self.model_xml[model_name]}'
            '</sdf>'
        )

    def _label_components(self) -> np.ndarray:
        labels = np.full(self.free.shape, -1, dtype=np.int32)
        next_label = 0
        rows, columns = self.free.shape
        for row, column in np.argwhere(self.free):
            if labels[row, column] >= 0:
                continue
            queue = deque([(int(row), int(column))])
            labels[row, column] = next_label
            while queue:
                current_row, current_column = queue.popleft()
                for delta_row, delta_column in (
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                    (-1, -1),
                    (-1, 1),
                    (1, -1),
                    (1, 1),
                ):
                    neighbor_row = current_row + delta_row
                    neighbor_column = current_column + delta_column
                    if not 0 <= neighbor_row < rows or not 0 <= neighbor_column < columns:
                        continue
                    if not self.free[neighbor_row, neighbor_column]:
                        continue
                    if labels[neighbor_row, neighbor_column] >= 0:
                        continue
                    # Prevent a diagonal from cutting through an obstacle corner.
                    if delta_row and delta_column:
                        if not self.free[current_row, neighbor_column]:
                            continue
                        if not self.free[neighbor_row, current_column]:
                            continue
                    labels[neighbor_row, neighbor_column] = next_label
                    queue.append((neighbor_row, neighbor_column))
            next_label += 1
        return labels

    def _coordinate(self, row: int, column: int) -> tuple[float, float]:
        return (
            float(self.x_coordinates[column]),
            float(self.y_coordinates[row]),
        )

    def grid_index(self, x: float, y: float) -> tuple[int, int] | None:
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        column = int(round((x - float(self.x_coordinates[0])) / self.resolution))
        row = int(round((y - float(self.y_coordinates[0])) / self.resolution))
        if not 0 <= row < self.free.shape[0] or not 0 <= column < self.free.shape[1]:
            return None
        return row, column

    def sample_episode(
        self,
        rng: np.random.Generator,
        min_goal_distance: float = 2.0,
        attempts: int = 500,
    ) -> EpisodeSample:
        candidates = np.argwhere(self.free)
        component_labels = np.array(
            [self.components[row, column] for row, column in candidates],
            dtype=np.int32,
        )
        usable_labels = {
            label for label, size in self.component_sizes.items() if size >= 2
        }
        valid_indices = np.array(
            [index for index, label in enumerate(component_labels) if label in usable_labels],
            dtype=np.int32,
        )
        if len(valid_indices) < 2:
            raise RuntimeError(f"not enough connected free space in {self.path.name}")

        for _ in range(attempts):
            start_candidate = int(rng.choice(valid_indices))
            start_row, start_column = candidates[start_candidate]
            label = component_labels[start_candidate]
            same_component = valid_indices[
                component_labels[valid_indices] == label
            ]
            goal_candidate = int(rng.choice(same_component))
            goal_row, goal_column = candidates[goal_candidate]
            start_x, start_y = self._coordinate(start_row, start_column)
            goal_x, goal_y = self._coordinate(goal_row, goal_column)
            if math.hypot(goal_x - start_x, goal_y - start_y) < min_goal_distance:
                continue
            distance_field = self.distance_field(goal_x, goal_y)
            shortest_path = float(distance_field[start_row, start_column])
            if not math.isfinite(shortest_path) or shortest_path < min_goal_distance:
                continue
            return EpisodeSample(
                start_x=start_x,
                start_y=start_y,
                start_yaw=float(rng.uniform(-math.pi, math.pi)),
                goal_x=goal_x,
                goal_y=goal_y,
                shortest_path=shortest_path,
            )
        raise RuntimeError(
            f"could not sample a connected start/goal pair in {self.path.name}"
        )

    def distance_field(self, goal_x: float, goal_y: float) -> np.ndarray:
        goal_index = self.grid_index(goal_x, goal_y)
        # Float64 avoids discarding valid queue entries after float32 rounding.
        distances = np.full(self.free.shape, np.inf, dtype=np.float64)
        if goal_index is None or not self.free[goal_index]:
            return distances

        distances[goal_index] = 0.0
        queue: list[tuple[float, int, int]] = [(0.0, goal_index[0], goal_index[1])]
        rows, columns = self.free.shape
        straight_cost = self.resolution
        diagonal_cost = self.resolution * math.sqrt(2.0)
        while queue:
            distance, row, column = heapq.heappop(queue)
            if distance > float(distances[row, column]) + 1e-9:
                continue
            for delta_row, delta_column in (
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ):
                neighbor_row = row + delta_row
                neighbor_column = column + delta_column
                if not 0 <= neighbor_row < rows or not 0 <= neighbor_column < columns:
                    continue
                if not self.free[neighbor_row, neighbor_column]:
                    continue
                if delta_row and delta_column:
                    if not self.free[row, neighbor_column]:
                        continue
                    if not self.free[neighbor_row, column]:
                        continue
                    step_cost = diagonal_cost
                else:
                    step_cost = straight_cost
                candidate = distance + step_cost
                if candidate + 1e-9 < float(distances[neighbor_row, neighbor_column]):
                    distances[neighbor_row, neighbor_column] = candidate
                    heapq.heappush(
                        queue,
                        (candidate, neighbor_row, neighbor_column),
                    )
        return distances

    def path_distance(
        self,
        x: float,
        y: float,
        distance_field: np.ndarray,
    ) -> float:
        index = self.grid_index(x, y)
        if index is None or not self.free[index]:
            return math.inf

        column_float = (
            x - float(self.x_coordinates[0])
        ) / self.resolution
        row_float = (
            y - float(self.y_coordinates[0])
        ) / self.resolution
        # Coordinates sampled from the float32 grid can differ from an exact
        # cell center by a few ulps.  Snap those values so sampled endpoints
        # retain an exact zero potential at the goal.
        if abs(column_float - round(column_float)) < 1e-4:
            column_float = float(round(column_float))
        if abs(row_float - round(row_float)) < 1e-4:
            row_float = float(round(row_float))
        column_low = max(
            0,
            min(int(math.floor(column_float)), self.free.shape[1] - 1),
        )
        row_low = max(
            0,
            min(int(math.floor(row_float)), self.free.shape[0] - 1),
        )
        column_high = min(column_low + 1, self.free.shape[1] - 1)
        row_high = min(row_low + 1, self.free.shape[0] - 1)
        column_fraction = min(max(column_float - column_low, 0.0), 1.0)
        row_fraction = min(max(row_float - row_low, 0.0), 1.0)
        samples = (
            (
                row_low,
                column_low,
                (1.0 - row_fraction) * (1.0 - column_fraction),
            ),
            (row_low, column_high, (1.0 - row_fraction) * column_fraction),
            (row_high, column_low, row_fraction * (1.0 - column_fraction)),
            (row_high, column_high, row_fraction * column_fraction),
        )
        weighted_distance = 0.0
        total_weight = 0.0
        for row, column, weight in samples:
            value = float(distance_field[row, column])
            if weight > 0.0 and self.free[row, column] and math.isfinite(value):
                weighted_distance += weight * value
                total_weight += weight
        if total_weight <= 0.0:
            return float(distance_field[index])
        return weighted_distance / total_weight

    def contains_safe_center(self, x: float, y: float) -> bool:
        """Return whether a pose center lies inside the inflated arena grid."""
        return self.grid_index(x, y) is not None

    def is_free_pose(self, x: float, y: float) -> bool:
        """Return whether a pose center has the configured obstacle clearance."""
        index = self.grid_index(x, y)
        return bool(index is not None and self.free[index])

    def contains(self, x: float, y: float) -> bool:
        min_x, max_x, min_y, max_y = self.bounds
        return min_x <= x <= max_x and min_y <= y <= max_y


def discover_worlds(worlds_directory: str | Path) -> tuple[Path, ...]:
    """Return ``mundo_N.world`` files in numeric order."""
    worlds_directory = Path(worlds_directory)

    def world_number(path: Path) -> int:
        match = re.fullmatch(r"mundo_(\d+)\.world", path.name)
        return int(match.group(1)) if match else 10**9

    paths = sorted(worlds_directory.glob("mundo_*.world"), key=world_number)
    return tuple(path.resolve() for path in paths if world_number(path) < 10**9)
