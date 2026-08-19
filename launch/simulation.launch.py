# ros2 launch martha simulation.launch.py gui:=false sim_speed_factor:=4.0

from pathlib import Path
import os
import tempfile

import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.actions import IncludeLaunchDescription
from launch.actions import OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from martha.simulation_speed import create_scaled_world
from martha.PPO.training_layout import parking_pose


def cleanup_scaled_world(_context, world_path):
    """Remove the generated SDF when its launch shuts down."""
    Path(world_path).unlink(missing_ok=True)
    return []


def create_controller_override(namespace):
    """Create namespaced odometry frame parameters for one controller."""
    prefix = f'{namespace}/' if namespace else ''
    document = {
        '/**/mecanum_drive_controller': {
            'ros__parameters': {
                'base_frame_id': f'{prefix}base_link',
                'odom_frame_id': f'{prefix}odom',
            },
        },
    }
    descriptor, filename = tempfile.mkstemp(
        prefix='martha_controller_',
        suffix='.yaml',
    )
    os.close(descriptor)
    path = Path(filename)
    with path.open('w', encoding='utf-8') as stream:
        yaml.safe_dump(document, stream, sort_keys=False)
    return path


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
            # Controllers need Gazebo update cycles while they are loaded.
            # The shared PPO coordinator pauses once all eight pipelines are
            # publishing; a standalone one-robot launch remains immediately
            # usable as before.
            'pause': 'false',
            'world': str(scaled_world),
        }.items(),
    )
    return [cleanup_handler, gazebo]


def launch_robots(context):
    """Create one root robot or a fully namespaced multi-robot fleet."""
    package_name = 'martha'
    try:
        robot_count = int(LaunchConfiguration('robot_count').perform(context))
    except ValueError as exc:
        raise RuntimeError('robot_count must be an integer') from exc
    if robot_count <= 0:
        raise RuntimeError('robot_count must be positive')

    actions = []
    entity_spawners = []
    startup_unpausers = []
    controller_spawners = []
    for index in range(robot_count):
        multi_robot = robot_count > 1
        namespace = f'martha_{index}' if multi_robot else ''
        frame_prefix = f'{namespace}/' if namespace else ''
        entity_name = namespace or 'robot'
        start_x, start_y, _ = (
            parking_pose(index, robot_count)
            if multi_robot
            else (0.0, 0.0, 0.0)
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
                ' robot_namespace:=',
                namespace,
                ' frame_prefix:=',
                frame_prefix,
            ]),
            value_type=str,
        )
        state_publisher = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            namespace=namespace,
            name='robot_state_publisher',
            parameters=[
                {'robot_description': robot_description},
                {'use_sim_time': True},
                {'frame_prefix': frame_prefix},
            ],
            output='screen',
        )
        spawner = Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            namespace=namespace,
            name='urdf_spawner',
            arguments=[
                '-topic',
                'robot_description',
                '-entity',
                entity_name,
                '-x',
                str(start_x),
                '-y',
                str(start_y),
                '-z',
                '0.5',
            ],
            output='screen',
        )
        controller_manager = f'/{namespace}/controller_manager' if namespace else (
            '/controller_manager'
        )
        controller_override = create_controller_override(namespace)
        controller_cleanup = RegisterEventHandler(
            OnShutdown(
                on_shutdown=[
                    OpaqueFunction(
                        function=cleanup_scaled_world,
                        kwargs={'world_path': str(controller_override)},
                    ),
                ],
            )
        )
        controller_spawner = Node(
            package='controller_manager',
            executable='spawner',
            namespace=namespace,
            name='controller_spawner',
            arguments=[
                'joint_state_broadcaster',
                'mecanum_drive_controller',
                '--controller-manager',
                controller_manager,
                '--param-file',
                str(controller_override),
                '--controller-manager-timeout',
                '60',
                '--service-call-timeout',
                '60',
                '--switch-timeout',
                '60',
            ],
            output='screen',
        )
        startup_unpauser = ExecuteProcess(
            cmd=[
                'bash',
                '-lc',
                'for attempt in 1 2 3; do gz world -p 0; sleep 0.5; done',
            ],
            name=f'unpause_after_{entity_name}',
            output='screen',
        )
        adapter = Node(
            package=package_name,
            executable='cmd_vel_to_twist_stamped',
            namespace=namespace,
            name='cmd_vel_to_twist_stamped',
            parameters=[{
                'use_sim_time': True,
                'input_topic': 'cmd_vel',
                'output_topic': 'mecanum_drive_controller/reference',
                'frame_id': f'{frame_prefix}base_link',
            }],
            output='screen',
        )
        ekf = Node(
            package='robot_localization',
            executable='ekf_node',
            namespace=namespace,
            name='ekf_filter_node',
            parameters=[
                PathJoinSubstitution([
                    FindPackageShare(package_name),
                    'config',
                    'ekf_rl_sim.yaml',
                ]),
                {
                    'map_frame': f'{frame_prefix}map',
                    'odom_frame': f'{frame_prefix}odom',
                    'base_link_frame': f'{frame_prefix}base_link',
                    'world_frame': f'{frame_prefix}odom',
                    'odom0': 'mecanum_drive_controller/odometry',
                    'imu0': 'imu/data',
                },
            ],
            output='screen',
        )
        actions.extend([
            controller_cleanup,
            state_publisher,
            adapter,
            ekf,
        ])
        entity_spawners.append(spawner)
        startup_unpausers.append(startup_unpauser)
        controller_spawners.append(controller_spawner)

    # Gazebo's spawn service and controller-manager switch callbacks are
    # intentionally serialized.  Parallel spawning can make plugins fetch a
    # neighbouring robot_description, while parallel controller switches can
    # starve the Gazebo update thread needed to complete those same switches.
    for index, (spawner, startup_unpauser, controller_spawner) in enumerate(
        zip(entity_spawners, startup_unpausers, controller_spawners)
    ):
        actions.append(
            RegisterEventHandler(
                OnProcessExit(
                    target_action=spawner,
                    on_exit=[startup_unpauser],
                )
            )
        )
        actions.append(
            RegisterEventHandler(
                OnProcessExit(
                    target_action=startup_unpauser,
                    on_exit=[controller_spawner],
                )
            )
        )
        if index + 1 < len(entity_spawners):
            actions.append(
                RegisterEventHandler(
                    OnProcessExit(
                        target_action=controller_spawner,
                        on_exit=[entity_spawners[index + 1]],
                    )
                )
            )
    actions.append(entity_spawners[0])
    return actions


def generate_launch_description():
    package_name = 'martha'

    default_world = PathJoinSubstitution([
        FindPackageShare(package_name),
        'worlds',
        'room.world',
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
    robot_count_argument = DeclareLaunchArgument(
        'robot_count',
        default_value='1',
        description='Cantidad de Marthas; valores mayores que uno usan namespaces',
    )

    # El controlador no publica TF (enable_odom_tf=false). El EKF anterior es
    # la unica autoridad dinamica para odom -> base_link.
    return LaunchDescription([
        world_argument,
        gui_argument,
        speed_argument,
        robot_count_argument,
        OpaqueFunction(function=launch_gazebo),
        OpaqueFunction(function=launch_robots),
    ])
