"""Shared-world layout and strict catalog-driven training episodes."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math
import os
from pathlib import Path
import tempfile
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from .world_map import EpisodeSample, TRAINING_WORLD_NAMES, WorldMap


WORLD_ORIGINS = {
    "four_rooms": (-30.0, 15.0),
    "hall": (0.0, 15.0),
    "multi": (30.0, 15.0),
    "roblab": (-30.0, -15.0),
    "room": (0.0, -15.0),
    "tube": (30.0, -15.0),
}
PARKING_Y = -43.0
PARKING_SPACING = 4.0
MIN_START_SEPARATION = 0.80


@dataclass(frozen=True)
class TrainingPoint:
    """One user-authored local point usable as a start or goal."""

    point_id: str
    x: float
    y: float
    yaw: float | None = None


def parking_pose(index: int, robot_count: int) -> tuple[float, float, float]:
    """Return a unique pose in the parking row below every arena."""
    if robot_count <= 0 or not 0 <= index < robot_count:
        raise ValueError("parking index must belong to a positive robot count")
    centered_index = index - 0.5 * (robot_count - 1)
    return centered_index * PARKING_SPACING, PARKING_Y, 0.0


def _finite_number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def load_training_points(path: str | Path) -> dict[str, tuple[TrainingPoint, ...]]:
    """Load the complete six-world point catalog without implicit fallbacks."""
    catalog_path = Path(path).expanduser().resolve()
    if not catalog_path.is_file():
        raise FileNotFoundError(f"training point catalog does not exist: {catalog_path}")
    with catalog_path.open(encoding="utf-8") as source:
        document = yaml.safe_load(source)
    if not isinstance(document, dict) or not isinstance(document.get("worlds"), dict):
        raise ValueError("training point catalog requires a 'worlds' mapping")

    raw_worlds = document["worlds"]
    unknown = sorted(set(raw_worlds) - set(TRAINING_WORLD_NAMES))
    missing = sorted(set(TRAINING_WORLD_NAMES) - set(raw_worlds))
    if unknown or missing:
        raise ValueError(
            "training point catalog world mismatch: "
            f"missing={missing}, unknown={unknown}"
        )

    catalog: dict[str, tuple[TrainingPoint, ...]] = {}
    for world_name in TRAINING_WORLD_NAMES:
        raw_world = raw_worlds[world_name]
        if not isinstance(raw_world, dict) or not isinstance(
            raw_world.get("points"), list
        ):
            raise ValueError(f"world {world_name} requires a points list")
        points = []
        identifiers: set[str] = set()
        for index, raw_point in enumerate(raw_world["points"]):
            label = f"{world_name}.points[{index}]"
            if not isinstance(raw_point, dict):
                raise ValueError(f"{label} must be a mapping")
            point_id = str(raw_point.get("id", "")).strip()
            if not point_id:
                raise ValueError(f"{label}.id cannot be empty")
            if point_id in identifiers:
                raise ValueError(f"world {world_name} repeats point id {point_id}")
            identifiers.add(point_id)
            yaw_value = raw_point.get("yaw")
            points.append(
                TrainingPoint(
                    point_id=point_id,
                    x=_finite_number(raw_point.get("x"), f"{label}.x"),
                    y=_finite_number(raw_point.get("y"), f"{label}.y"),
                    yaw=(
                        None
                        if yaw_value is None
                        else _finite_number(yaw_value, f"{label}.yaw")
                    ),
                )
            )
        catalog[world_name] = tuple(points)
    return catalog


def _starts_are_separated(points: list[TrainingPoint]) -> bool:
    for index, first in enumerate(points):
        for second in points[index + 1:]:
            if math.hypot(first.x - second.x, first.y - second.y) < (
                MIN_START_SEPARATION
            ):
                return False
    return True


def _valid_goal_candidates(
    world_map: WorldMap,
    points: tuple[TrainingPoint, ...],
    start: TrainingPoint,
    min_goal_distance: float,
) -> list[TrainingPoint]:
    start_index = world_map.grid_index(start.x, start.y)
    if start_index is None or not world_map.free[start_index]:
        return []
    start_component = world_map.components[start_index]
    candidates = []
    for goal in points:
        if goal.point_id == start.point_id:
            continue
        goal_index = world_map.grid_index(goal.x, goal.y)
        if goal_index is None or not world_map.free[goal_index]:
            continue
        # Same component proves connectivity.  Geodesic distance can never be
        # shorter than Euclidean distance, so this is also a strict and much
        # cheaper minimum-distance check for the full catalog.
        if (
            world_map.components[goal_index] == start_component
            and math.hypot(goal.x - start.x, goal.y - start.y)
            >= min_goal_distance
        ):
            candidates.append(goal)
    return candidates


def validate_training_points(
    catalog: dict[str, tuple[TrainingPoint, ...]],
    world_maps: tuple[WorldMap, ...],
    *,
    robot_count: int,
    min_goal_distance: float,
) -> None:
    """Reject incomplete, occupied or unusable catalogs before Gazebo starts."""
    maps_by_name = {world.world_name: world for world in world_maps}
    for world_name in TRAINING_WORLD_NAMES:
        world_map = maps_by_name.get(world_name)
        if world_map is None:
            raise ValueError(f"world geometry is unavailable for {world_name}")
        points = catalog[world_name]
        if len(points) < robot_count:
            raise ValueError(
                f"world {world_name} needs at least {robot_count} points; "
                f"found {len(points)}"
            )
        occupied = [
            point.point_id
            for point in points
            if not world_map.is_free_pose(point.x, point.y)
        ]
        if occupied:
            raise ValueError(
                f"world {world_name} has points outside safe free space: {occupied}"
            )
        without_goals = [
            point.point_id
            for point in points
            if not _valid_goal_candidates(
                world_map,
                points,
                point,
                min_goal_distance,
            )
        ]
        if without_goals:
            raise ValueError(
                f"world {world_name} has starts without a distinct connected "
                f"goal at the minimum distance: {without_goals}"
            )

        ordered = list(points)

        def has_start_set(chosen: list[TrainingPoint], cursor: int) -> bool:
            if len(chosen) == robot_count:
                return True
            remaining_needed = robot_count - len(chosen)
            if len(ordered) - cursor < remaining_needed:
                return False
            for candidate_index in range(cursor, len(ordered)):
                candidate = ordered[candidate_index]
                if _starts_are_separated([*chosen, candidate]) and has_start_set(
                    [*chosen, candidate], candidate_index + 1
                ):
                    return True
            return False

        if not has_start_set([], 0):
            raise ValueError(
                f"world {world_name} cannot place {robot_count} starts at least "
                f"{MIN_START_SEPARATION:.2f} m apart"
            )


def sample_round_episodes(
    world_map: WorldMap,
    points: tuple[TrainingPoint, ...],
    origin: tuple[float, float],
    *,
    robot_count: int,
    min_goal_distance: float,
    rng: np.random.Generator,
    distance_field_cache: dict[str, np.ndarray] | None = None,
) -> tuple[EpisodeSample, ...]:
    """Sample one collision-free group from a validated point catalog."""
    starts: list[TrainingPoint] | None = None
    for _ in range(500):
        indices = rng.choice(len(points), size=robot_count, replace=False)
        candidate_starts = [points[int(index)] for index in indices]
        if _starts_are_separated(candidate_starts):
            starts = candidate_starts
            break
    if starts is None:
        raise RuntimeError(
            f"could not sample {robot_count} separated starts in {world_map.world_name}"
        )

    offset_x, offset_y = origin
    fields = {} if distance_field_cache is None else distance_field_cache
    episodes = []
    for start in starts:
        candidates = _valid_goal_candidates(
            world_map,
            points,
            start,
            min_goal_distance,
        )
        if not candidates:
            raise RuntimeError(
                f"point {start.point_id} has no valid goal in {world_map.world_name}"
            )
        goal = candidates[int(rng.integers(len(candidates)))]
        distance_field = fields.get(goal.point_id)
        if distance_field is None:
            distance_field = world_map.distance_field(goal.x, goal.y)
            fields[goal.point_id] = distance_field
        shortest_path = world_map.path_distance(
            start.x,
            start.y,
            distance_field,
        )
        if not math.isfinite(shortest_path) or shortest_path < min_goal_distance:
            raise RuntimeError(
                f"invalid cached route {start.point_id}->{goal.point_id} "
                f"in {world_map.world_name}"
            )
        yaw = start.yaw
        if yaw is None:
            yaw = float(rng.uniform(-math.pi, math.pi))
        episodes.append(
            EpisodeSample(
                start_x=start.x + offset_x,
                start_y=start.y + offset_y,
                start_yaw=yaw,
                goal_x=goal.x + offset_x,
                goal_y=goal.y + offset_y,
                shortest_path=shortest_path,
            )
        )
    return tuple(episodes)


def create_combined_training_world(
    world_paths: tuple[Path, ...],
    *,
    directory: str | Path | None = None,
) -> Path:
    """Create one temporary SDF containing all six offset training arenas."""
    paths_by_name = {path.stem: path for path in world_paths}
    missing = [name for name in TRAINING_WORLD_NAMES if name not in paths_by_name]
    if missing:
        raise FileNotFoundError(f"training worlds are missing: {missing}")

    first_world = ET.parse(paths_by_name[TRAINING_WORLD_NAMES[0]]).getroot().find(
        "world"
    )
    if first_world is None:
        raise ValueError("canonical training SDF has no world element")
    root = ET.Element("sdf", {"version": "1.6"})
    combined = ET.SubElement(root, "world", {"name": "martha_training_islands"})
    for tag in ("include", "physics", "scene"):
        for element in first_world.findall(tag):
            combined.append(copy.deepcopy(element))
    state_plugin = ET.SubElement(
        combined,
        "plugin",
        {"name": "gazebo_ros_state", "filename": "libgazebo_ros_state.so"},
    )
    ros = ET.SubElement(state_plugin, "ros")
    ET.SubElement(ros, "namespace").text = "/gazebo"
    ET.SubElement(state_plugin, "update_rate").text = "100.0"

    for world_name in TRAINING_WORLD_NAMES:
        source_world = ET.parse(paths_by_name[world_name]).getroot().find("world")
        if source_world is None:
            raise ValueError(f"{world_name}.world has no world element")
        offset_x, offset_y = WORLD_ORIGINS[world_name]
        for source_model in source_world.findall("model"):
            if source_model.get("name") == "goal_point":
                continue
            model = copy.deepcopy(source_model)
            source_name = model.get("name")
            if not source_name:
                raise ValueError(
                    f"{world_name}.world contains a model without a name"
                )
            # Model names are global within one SDF world.  Prefix common
            # names such as wall_left so the six islands can coexist.
            model.set("name", f"{world_name}__{source_name}")
            pose = model.find("pose")
            if pose is None:
                pose = ET.Element("pose")
                model.insert(0, pose)
                values = [0.0] * 6
            else:
                values = [float(value) for value in (pose.text or "").split()]
                values.extend([0.0] * (6 - len(values)))
            values[0] += offset_x
            values[1] += offset_y
            pose.text = " ".join(f"{value:.9g}" for value in values[:6])
            combined.append(model)

    target_directory = None if directory is None else str(directory)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="martha_training_islands_",
        suffix=".world",
        dir=target_directory,
    )
    os.close(descriptor)
    target = Path(temporary_name)
    try:
        ET.ElementTree(root).write(target, encoding="utf-8", xml_declaration=True)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return target


def shuffled_world_index(seed: int, round_index: int, world_count: int) -> int:
    """Choose one map from reproducible shuffled cycles without repetition."""
    if round_index < 0 or world_count <= 0:
        raise ValueError("round index and world count must be valid")
    cycle_index, position = divmod(round_index, world_count)
    cycle_rng = np.random.default_rng(np.random.SeedSequence([seed, 2, cycle_index]))
    return int(cycle_rng.permutation(world_count)[position])
