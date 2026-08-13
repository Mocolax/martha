"""Run the same PPO policy on either Gazebo or the physical robot."""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.actions import OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def validate_checkpoint(context):
    """Fail before bringup when the requested policy file is unavailable."""
    checkpoint = LaunchConfiguration("checkpoint").perform(context).strip()
    if not checkpoint:
        raise RuntimeError("checkpoint:=/absolute/path/to/best_model.pt is required")
    if not Path(checkpoint).is_absolute():
        raise RuntimeError("checkpoint must be an absolute path")
    if not Path(checkpoint).is_file():
        raise RuntimeError(f"PPO checkpoint does not exist: {checkpoint}")
    return []


def launch_policy(context):
    """Create a policy with only the fault source owned by its backend."""
    mode = LaunchConfiguration("mode").perform(context).strip()
    if mode not in {"sim", "hardware"}:
        raise RuntimeError("mode must be 'sim' or 'hardware'")
    return [
        Node(
            package="martha",
            executable="ppo_policy",
            name="ppo_policy",
            parameters=[{
                "checkpoint": LaunchConfiguration("checkpoint"),
                "device": LaunchConfiguration("device"),
                "use_sim_time": mode == "sim",
                "fault_topic": (
                    "" if mode == "sim" else "/hardware/motor_fault"
                ),
            }],
            output="screen",
        )
    ]


def generate_launch_description():
    """Compose common bringup and the backend-independent policy node."""
    package_name = "martha"
    default_world = PathJoinSubstitution([
        FindPackageShare(package_name),
        "worlds",
        "mundo_1.world",
    ])
    arguments = [
        DeclareLaunchArgument(
            "checkpoint",
            default_value="",
            description="Absolute path to a versioned Martha PPO checkpoint",
        ),
        DeclareLaunchArgument("mode", default_value="sim"),
        DeclareLaunchArgument("world", default_value=default_world),
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("sim_speed_factor", default_value="1.0"),
        DeclareLaunchArgument("port", default_value="/dev/ttyUSB0"),
        DeclareLaunchArgument("lidar_port", default_value="/dev/rplidar"),
        DeclareLaunchArgument("lidar_frame", default_value="lidar"),
        DeclareLaunchArgument(
            "lidar_scan_mode",
            default_value="Sensitivity",
        ),
        DeclareLaunchArgument("start_lidar", default_value="true"),
        DeclareLaunchArgument("mapping", default_value="false"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("device", default_value="auto"),
    ]

    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare(package_name),
                "launch",
                "bringup.launch.py",
            ])
        ),
        launch_arguments={
            "mode": LaunchConfiguration("mode"),
            "world": LaunchConfiguration("world"),
            "gui": LaunchConfiguration("gui"),
            "sim_speed_factor": LaunchConfiguration("sim_speed_factor"),
            "port": LaunchConfiguration("port"),
            "lidar_port": LaunchConfiguration("lidar_port"),
            "lidar_frame": LaunchConfiguration("lidar_frame"),
            "lidar_scan_mode": LaunchConfiguration("lidar_scan_mode"),
            "start_lidar": LaunchConfiguration("start_lidar"),
            "mapping": LaunchConfiguration("mapping"),
            "rviz": LaunchConfiguration("rviz"),
        }.items(),
    )
    return LaunchDescription([
        *arguments,
        OpaqueFunction(function=validate_checkpoint),
        bringup,
        OpaqueFunction(function=launch_policy),
    ])
