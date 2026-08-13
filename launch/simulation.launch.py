# ros2 launch martha simulation.launch.py gui:=false sim_speed_factor:=4.0

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.actions import OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from martha.simulation_speed import create_scaled_world


def cleanup_scaled_world(_context, world_path):
    """Remove the generated SDF when its launch shuts down."""
    Path(world_path).unlink(missing_ok=True)
    return []


def launch_gazebo(context):
    """Launch Gazebo with a temporary world at the requested speed factor."""
    source_world = LaunchConfiguration('world').perform(context)
    speed_factor = LaunchConfiguration('sim_speed_factor').perform(context)
    scaled_world = create_scaled_world(source_world, speed_factor)
    cleanup_handler = RegisterEventHandler(
        OnShutdown(
            on_shutdown=[
                OpaqueFunction(
                    function=cleanup_scaled_world,
                    kwargs={'world_path': str(scaled_world)},
                ),
            ],
        )
    )
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py',
            ])
        ),
        launch_arguments={
            'gui': LaunchConfiguration('gui'),
            'pause': 'true',
            'world': str(scaled_world),
        }.items(),
    )
    return [cleanup_handler, gazebo]


def generate_launch_description():
    package_name = 'martha'

    default_world = PathJoinSubstitution([
        FindPackageShare(package_name),
        'worlds',
        'mundo_1.world',
    ])
    world_argument = DeclareLaunchArgument(
        'world',
        default_value=default_world,
        description='Archivo world que cargara Gazebo',
    )
    gui_argument = DeclareLaunchArgument(
        'gui',
        default_value='true',
        description='Inicia la interfaz grafica de Gazebo',
    )
    speed_argument = DeclareLaunchArgument(
        'sim_speed_factor',
        default_value='1.0',
        description='Factor objetivo de tiempo simulado; rango (0, 20]',
    )

    robot_description = ParameterValue(
        Command([
            FindExecutable(name='xacro'),
            ' ',
            PathJoinSubstitution([
                FindPackageShare(package_name),
                'urdf',
                'learning.xacro',
            ]),
        ]),
        value_type=str,
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'robot_description': robot_description},
            {'use_sim_time': True},
        ],
        output='screen',
    )

    robot_spawner = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='urdf_spawner',
        arguments=[
            '-topic',
            '/robot_description',
            '-entity',
            'robot',
            '-z',
            '0.5',
            '-unpause',
        ],
        output='screen',
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen',
    )
    mecanum_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['mecanum_drive_controller'],
        output='screen',
    )
    controller_spawners = RegisterEventHandler(
        OnProcessExit(
            target_action=robot_spawner,
            on_exit=[
                joint_state_broadcaster_spawner,
                mecanum_controller_spawner,
            ],
        )
    )

    cmd_vel_adapter = Node(
        package=package_name,
        executable='cmd_vel_to_twist_stamped',
        name='cmd_vel_to_twist_stamped',
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    ekf_filter = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare(package_name),
                'config',
                'ekf_rl_sim.yaml',
            ]),
        ],
        output='screen',
    )

    # El controlador no publica TF (enable_odom_tf=false). El EKF anterior es
    # la unica autoridad dinamica para odom -> base_link.
    return LaunchDescription([
        world_argument,
        gui_argument,
        speed_argument,
        OpaqueFunction(function=launch_gazebo),
        robot_state_publisher,
        robot_spawner,
        controller_spawners,
        cmd_vel_adapter,
        ekf_filter,
    ])
