"""
Geometry helpers for the Gazebo worlds used by PPO.

This module deliberately has no ROS or Gymnasium dependency.  It parses the
simple SDF arenas stored in ``worlds/`` and builds an inflated 2-D free-space
grid.  The same grid is used to sample valid start/goal pairs and to compute a
geodesic potential for reward shaping.
"""

from __future__ import annotations

from collections import deque
import copy
from dataclasses import dataclass
import heapq
import math
from pathlib import Path
from typing import Iterable, Protocol
import xml.etree.ElementTree as ET

import numpy as np


TRAINING_WORLD_NAMES = (
    "four_rooms",
    "hall",
    "multi",
    "roblab",
    "room",
    "tube",
)
LOCAL_ARENA_BOUNDS = (-10.0, 10.0, -10.0, 10.0)


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
class CircleObstacle:
    """A cylindrical collision projected onto the world XY plane."""

    name: str
    x: float
    y: float
    radius: float

    @property
    def projected_half_x(self) -> float:
        return self.radius

    @property
    def projected_half_y(self) -> float:
        return self.radius

    def contains(self, x: np.ndarray, y: np.ndarray, margin: float) -> np.ndarray:
        """Return a mask for points inside this circle inflated by ``margin``."""
        radius = self.radius + margin
        return (x - self.x) ** 2 + (y - self.y) ** 2 <= radius**2


class Obstacle(Protocol):
    """Geometry contract used to rasterize supported SDF collisions."""

    name: str
    x: float
    y: float

    def contains(self, x: np.ndarray, y: np.ndarray, margin: float) -> np.ndarray:
        """Return an occupancy mask."""


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
        obstacles: Iterable[Obstacle],
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
        bounds: tuple[float, float, float, float] = LOCAL_ARENA_BOUNDS,
    ) -> "WorldMap":
        path = Path(path).resolve()
        root = ET.parse(path).getroot()
        world = root.find("world")
        if world is None:
            raise ValueError(f"{path} does not contain an SDF world")

        obstacles: list[Obstacle] = []
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
                    collision_name = collision.get(
                        "name",
                        f"collision_{link_index}_{collision_index}",
                    )
                    size_element = collision.find("geometry/box/size")
                    collision_pose = _compose_pose(
                        link_pose,
                        _pose_2d(collision.findtext("pose")),
                    )
                    obstacle_name = f"{model_name}/{collision_name}"
                    if size_element is not None and size_element.text:
                        sizes = [float(value) for value in size_element.text.split()]
                        if len(sizes) < 2:
                            continue
                        obstacles.append(
                            BoxObstacle(
                                name=obstacle_name,
                                x=collision_pose[0],
                                y=collision_pose[1],
                                size_x=sizes[0],
                                size_y=sizes[1],
                                yaw=collision_pose[2],
                            )
                        )
                        continue
                    radius_element = collision.find("geometry/cylinder/radius")
                    if radius_element is not None and radius_element.text:
                        obstacles.append(
                            CircleObstacle(
                                name=obstacle_name,
                                x=collision_pose[0],
                                y=collision_pose[1],
                                radius=float(radius_element.text),
                            )
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

    def translated(self, offset_x: float, offset_y: float) -> "WorldMap":
        """Return the same occupancy geometry translated into a shared world."""
        if not math.isfinite(offset_x) or not math.isfinite(offset_y):
            raise ValueError("world translation offsets must be finite")
        translated_obstacles: list[Obstacle] = []
        for obstacle in self.obstacles:
            if isinstance(obstacle, BoxObstacle):
                translated_obstacles.append(
                    BoxObstacle(
                        name=obstacle.name,
                        x=obstacle.x + offset_x,
                        y=obstacle.y + offset_y,
                        size_x=obstacle.size_x,
                        size_y=obstacle.size_y,
                        yaw=obstacle.yaw,
                    )
                )
            else:
                translated_obstacles.append(
                    CircleObstacle(
                        name=obstacle.name,
                        x=obstacle.x + offset_x,
                        y=obstacle.y + offset_y,
                        radius=obstacle.radius,
                    )
                )
        # A rigid translation cannot change occupancy or connectivity. Rebuilding
        # the float32 grid with ``np.arange`` at a different origin can move a
        # boundary cell across an inflated obstacle by a few ulps. That made a
        # catalog point valid in its local map but occupied in the shared map.
        translated = copy.copy(self)
        min_x, max_x, min_y, max_y = self.bounds
        translated.bounds = (
            min_x + offset_x,
            max_x + offset_x,
            min_y + offset_y,
            max_y + offset_y,
        )
        translated.obstacles = tuple(translated_obstacles)
        translated.x_coordinates = (
            self.x_coordinates.astype(np.float64) + float(offset_x)
        )
        translated.y_coordinates = (
            self.y_coordinates.astype(np.float64) + float(offset_y)
        )
        translated.free = self.free.copy()
        translated.components = self.components.copy()
        translated.component_sizes = dict(self.component_sizes)
        return translated

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

    def geodesic_direction(
        self,
        x: float,
        y: float,
        distance_field: np.ndarray,
        *,
        lookahead: float | None = None,
    ) -> tuple[float, float] | None:
        """Return a unit world-frame direction descending the path field."""
        index = self.grid_index(x, y)
        if index is None or not self.free[index]:
            return None
        if distance_field.shape != self.free.shape:
            raise ValueError("distance field shape does not match the map")
        if lookahead is None:
            lookahead = self.resolution
        steps = max(1, int(math.ceil(float(lookahead) / self.resolution)))
        row, column = index
        rows, columns = self.free.shape
        for _ in range(steps):
            current_distance = float(distance_field[row, column])
            best = (row, column)
            best_distance = current_distance
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
                candidate = float(distance_field[neighbor_row, neighbor_column])
                if candidate + 1e-9 < best_distance:
                    best = (neighbor_row, neighbor_column)
                    best_distance = candidate
            if best == (row, column):
                break
            row, column = best
        target_x = float(self.x_coordinates[column])
        target_y = float(self.y_coordinates[row])
        delta_x = target_x - float(x)
        delta_y = target_y - float(y)
        norm = math.hypot(delta_x, delta_y)
        if norm <= 1e-6 or not math.isfinite(norm):
            return None
        return delta_x / norm, delta_y / norm

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
    """Return the six canonical training worlds in stable semantic order."""
    worlds_directory = Path(worlds_directory)
    return tuple(
        path.resolve()
        for name in TRAINING_WORLD_NAMES
        if (path := worlds_directory / f"{name}.world").is_file()
    )
