"""Show one selected PPO arena and all of its configured spawn points."""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.actions import LogInfo, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from martha.PPO.training_layout import (
    create_training_points_visualization_world,
    load_training_points,
)
from martha.PPO.world_map import discover_worlds


def cleanup_visualization_world(_context, world_path):
    """Remove the generated marker world when Gazebo closes."""
    Path(world_path).unlink(missing_ok=True)
    return []


def launch_training_points(context):
    """Generate the current catalog visualization and open it in Gazebo."""
    package_share = Path(FindPackageShare("martha").perform(context))
    points_path = Path(
        LaunchConfiguration("points").perform(context)
    ).expanduser().resolve()
    map_name = LaunchConfiguration("map").perform(context).strip()
    catalog = load_training_points(points_path)
    world_path = create_training_points_visualization_world(
        discover_worlds(package_share / "worlds"),
        catalog,
        selected_world=map_name,
    )

    cleanup = RegisterEventHandler(
        OnShutdown(
            on_shutdown=[
                OpaqueFunction(
                    function=cleanup_visualization_world,
                    kwargs={"world_path": str(world_path)},
                )
            ]
        )
    )
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("gazebo_ros"), "launch", "gazebo.launch.py"]
            )
        ),
        launch_arguments={
            "gui": LaunchConfiguration("gui"),
            "pause": "true",
            "world": str(world_path),
        }.items(),
    )
    point_count = len(catalog[map_name])
    return [
        LogInfo(
            msg=(
                f"Mostrando el mapa {map_name} con {point_count} puntos "
                f"desde {points_path}"
            )
        ),
        cleanup,
        gazebo,
    ]


def generate_launch_description():
    """Return the standalone training-point visualization launch."""
    default_points = PathJoinSubstitution(
        [FindPackageShare("martha"), "config", "training_points.yaml"]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "map",
                default_value="room",
                description=(
                    "Mapa: four_rooms, hall, multi, roblab, room o tube"
                ),
            ),
            DeclareLaunchArgument(
                "points",
                default_value=default_points,
                description="Catalogo YAML de puntos que se mostrara",
            ),
            DeclareLaunchArgument(
                "gui",
                default_value="true",
                description="Inicia la interfaz grafica de Gazebo",
            ),
            OpaqueFunction(function=launch_training_points),
        ]
    )
