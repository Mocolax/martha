"""Shared-world layout and strict catalog-driven training episodes."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math
import os
from pathlib import Path
import re
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
TRAINING_POINT_COLORS = {
    "four_rooms": (1.0, 0.15, 0.10, 1.0),
    "hall": (0.10, 1.0, 0.20, 1.0),
    "multi": (0.10, 0.35, 1.0, 1.0),
    "roblab": (1.0, 0.80, 0.05, 1.0),
    "room": (1.0, 0.10, 0.85, 1.0),
    "tube": (0.05, 0.95, 1.0, 1.0),
}


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
    max_goal_distance: float | None = None,
    rng: np.random.Generator,
    distance_field_cache: dict[str, np.ndarray] | None = None,
) -> tuple[EpisodeSample, ...]:
    """Sample one collision-free group from a validated point catalog."""
    if max_goal_distance is not None:
        if not math.isfinite(max_goal_distance):
            raise ValueError("max_goal_distance must be finite")
        if max_goal_distance < min_goal_distance:
            raise ValueError(
                "max_goal_distance cannot be smaller than min_goal_distance"
            )
    fields = {} if distance_field_cache is None else distance_field_cache
    routes_by_start: dict[
        str,
        list[tuple[TrainingPoint, float]],
    ] = {}
    for start in points:
        routes = []
        for candidate in _valid_goal_candidates(
            world_map,
            points,
            start,
            min_goal_distance,
        ):
            distance_field = fields.get(candidate.point_id)
            if distance_field is None:
                distance_field = world_map.distance_field(
                    candidate.x,
                    candidate.y,
                )
                fields[candidate.point_id] = distance_field
            shortest_path = world_map.path_distance(
                start.x,
                start.y,
                distance_field,
            )
            if (
                math.isfinite(shortest_path)
                and shortest_path >= min_goal_distance
                and (
                    max_goal_distance is None
                    or shortest_path <= max_goal_distance
                )
            ):
                routes.append((candidate, shortest_path))
        if routes:
            routes_by_start[start.point_id] = routes

    eligible_starts = [
        point for point in points if point.point_id in routes_by_start
    ]
    if len(eligible_starts) < robot_count:
        raise RuntimeError(
            f"world {world_map.world_name} has only {len(eligible_starts)} "
            f"starts with a route in [{min_goal_distance}, "
            f"{max_goal_distance}] m; need {robot_count}"
        )

    starts: list[TrainingPoint] | None = None
    for _ in range(500):
        indices = rng.choice(
            len(eligible_starts),
            size=robot_count,
            replace=False,
        )
        candidate_starts = [
            eligible_starts[int(index)] for index in indices
        ]
        if _starts_are_separated(candidate_starts):
            starts = candidate_starts
            break
    if starts is None:
        raise RuntimeError(
            f"could not sample {robot_count} separated starts in {world_map.world_name}"
        )

    offset_x, offset_y = origin
    episodes = []
    for start in starts:
        routes = routes_by_start[start.point_id]
        goal, shortest_path = routes[int(rng.integers(len(routes)))]
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


def _marker_identifier(value: str) -> str:
    """Return a readable SDF-safe fragment for a point marker name."""
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")
    return cleaned or "unnamed"


def _append_training_point_markers(
    world: ET.Element,
    catalog: dict[str, tuple[TrainingPoint, ...]],
    world_names: tuple[str, ...],
    origins: dict[str, tuple[float, float]],
) -> None:
    """Append collision-free visual markers at the requested catalog points."""
    world.append(
        ET.Comment(" Visual-only training point markers; colors are one per map. ")
    )
    for world_name in world_names:
        offset_x, offset_y = origins[world_name]
        rgba = TRAINING_POINT_COLORS[world_name]
        color = " ".join(f"{component:.6g}" for component in rgba)
        for index, point in enumerate(catalog[world_name], start=1):
            marker = ET.SubElement(
                world,
                "model",
                {
                    "name": (
                        f"training_point__{world_name}__{index:02d}__"
                        f"{_marker_identifier(point.point_id)}"
                    )
                },
            )
            ET.SubElement(marker, "static").text = "true"
            ET.SubElement(marker, "pose").text = (
                f"{point.x + offset_x:.9g} "
                f"{point.y + offset_y:.9g} 0.6 0 0 0"
            )
            link = ET.SubElement(marker, "link", {"name": "marker"})
            visual = ET.SubElement(link, "visual", {"name": "spawn_point"})
            ET.SubElement(visual, "cast_shadows").text = "false"
            geometry = ET.SubElement(visual, "geometry")
            cylinder = ET.SubElement(geometry, "cylinder")
            ET.SubElement(cylinder, "radius").text = "0.16"
            ET.SubElement(cylinder, "length").text = "1.2"
            material = ET.SubElement(visual, "material")
            ET.SubElement(material, "ambient").text = color
            ET.SubElement(material, "diffuse").text = color
            ET.SubElement(material, "emissive").text = color


def create_training_points_visualization_world(
    world_paths: tuple[Path, ...],
    catalog: dict[str, tuple[TrainingPoint, ...]],
    *,
    selected_world: str | None = None,
    directory: str | Path | None = None,
) -> Path:
    """Create one selected map, or all maps, with catalog points highlighted."""
    missing = [name for name in TRAINING_WORLD_NAMES if name not in catalog]
    if missing:
        raise ValueError(f"training point catalog is missing worlds: {missing}")

    paths_by_name = {path.stem: path for path in world_paths}
    temporary_source: Path | None = None
    if selected_world is None:
        temporary_source = create_combined_training_world(
            world_paths,
            directory=directory,
        )
        tree = ET.parse(temporary_source)
        shown_worlds = TRAINING_WORLD_NAMES
        origins = WORLD_ORIGINS
        camera_height = 85.0
    else:
        selected_world = str(selected_world).strip()
        if selected_world not in TRAINING_WORLD_NAMES:
            choices = ", ".join(TRAINING_WORLD_NAMES)
            raise ValueError(f"map must be one of: {choices}")
        source_path = paths_by_name.get(selected_world)
        if source_path is None:
            raise FileNotFoundError(f"world file is unavailable for {selected_world}")
        tree = ET.parse(source_path)
        shown_worlds = (selected_world,)
        origins = {selected_world: (0.0, 0.0)}
        camera_height = 34.0

    target: Path | None = None
    try:
        world = tree.getroot().find("world")
        if world is None:
            raise ValueError("training point visualization SDF has no world element")
        if selected_world is not None:
            world.set("name", f"{selected_world}_training_points")
            for element in list(world.findall("model")):
                if element.get("name") == "goal_point":
                    world.remove(element)
            for tag in ("gui", "state"):
                for element in list(world.findall(tag)):
                    world.remove(element)

        _append_training_point_markers(world, catalog, shown_worlds, origins)

        gui = ET.SubElement(world, "gui", {"fullscreen": "0"})
        camera = ET.SubElement(gui, "camera", {"name": "training_points_camera"})
        ET.SubElement(camera, "pose").text = (
            f"0 0 {camera_height:.9g} 0 1.5708 0"
        )
        ET.SubElement(camera, "view_controller").text = "orbit"
        ET.SubElement(camera, "projection_type").text = "perspective"

        target_directory = None if directory is None else str(directory)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="martha_training_points_",
            suffix=".world",
            dir=target_directory,
        )
        os.close(descriptor)
        target = Path(temporary_name)
        tree.write(target, encoding="utf-8", xml_declaration=True)
        return target
    except Exception:
        if target is not None:
            target.unlink(missing_ok=True)
        raise
    finally:
        if temporary_source is not None:
            temporary_source.unlink(missing_ok=True)


def shuffled_world_index(seed: int, round_index: int, world_count: int) -> int:
    """Choose one map from reproducible shuffled cycles without repetition."""
    if round_index < 0 or world_count <= 0:
        raise ValueError("round index and world count must be valid")
    cycle_index, position = divmod(round_index, world_count)
    cycle_rng = np.random.default_rng(np.random.SeedSequence([seed, 2, cycle_index]))
    return int(cycle_rng.permutation(world_count)[position])
