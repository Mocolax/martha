"""Compatibility alias for simulation plus odometry RViz."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_name = "martha"
    default_world = PathJoinSubstitution([
        FindPackageShare(package_name),
        "worlds",
        "mundo_2.world",
    ])
    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare(package_name),
                "launch",
                "bringup.launch.py",
            ])
        ),
        launch_arguments={
            "mode": "sim",
            "gui": LaunchConfiguration("gui"),
            "world": LaunchConfiguration("world"),
            "mapping": "false",
            "rviz": "true",
        }.items(),
    )
    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("world", default_value=default_world),
        bringup,
    ])
