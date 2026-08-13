"""Create temporary Gazebo worlds with a bounded real-time target."""

from __future__ import annotations

import math
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET


MAX_SIM_SPEED_FACTOR = 20.0


def validate_sim_speed_factor(value: object) -> float:
    """Return a finite factor in the supported faster-than-real-time range."""
    try:
        factor = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("sim_speed_factor must be a number") from exc
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError(
            "sim_speed_factor must be finite and greater than zero"
        )
    if factor > MAX_SIM_SPEED_FACTOR:
        raise ValueError(
            f"sim_speed_factor cannot exceed {MAX_SIM_SPEED_FACTOR:g}"
        )
    return factor


def create_scaled_world(
    source: str | Path,
    speed_factor: object,
    *,
    directory: str | Path | None = None,
) -> Path:
    """Copy an SDF world and set its target real-time factor."""
    factor = validate_sim_speed_factor(speed_factor)
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Gazebo world does not exist: {source_path}")

    tree = ET.parse(source_path)
    physics = tree.getroot().find("./world/physics")
    if physics is None:
        raise ValueError(
            f"Gazebo world has no <physics> element: {source_path}"
        )
    max_step_element = physics.find("max_step_size")
    if max_step_element is None or max_step_element.text is None:
        raise ValueError("Gazebo world physics has no max_step_size")
    try:
        max_step_size = float(max_step_element.text)
    except ValueError as exc:
        raise ValueError("Gazebo max_step_size must be numeric") from exc
    if not math.isfinite(max_step_size) or max_step_size <= 0.0:
        raise ValueError("Gazebo max_step_size must be finite and positive")

    update_rate = factor / max_step_size
    update_rate_element = physics.find("real_time_update_rate")
    if update_rate_element is None:
        update_rate_element = ET.SubElement(physics, "real_time_update_rate")
    update_rate_element.text = f"{update_rate:.12g}"
    factor_element = physics.find("real_time_factor")
    if factor_element is None:
        factor_element = ET.SubElement(physics, "real_time_factor")
    factor_element.text = f"{factor:.12g}"

    target_directory = None if directory is None else str(directory)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="martha_scaled_",
        suffix=".world",
        dir=target_directory,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        tree.write(temporary_path, encoding="utf-8", xml_declaration=True)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path
