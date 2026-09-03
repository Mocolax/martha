# ros2 launch martha simulation.launch.py gui:=false sim_speed_factor:=4.0

from pathlib import Path
import os
import tempfile

import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
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


def robot_description_topic(namespace):
    """
    Return the unambiguous description topic used to spawn one robot.

    The factory clients remain at the ROS root, so every one receives an
    absolute topic instead of relying on relative topic resolution. Each
    plugin carries a node-targeted ROS namespace remap in the description,
    avoiding Gazebo Classic's process-wide namespace rewrite for dynamically
    spawned fleets.
    """
    clean_namespace = namespace.strip('/')
    if not clean_namespace:
        return '/robot_description'
    return f'/{clean_namespace}/robot_description'


def parse_launch_boolean(value, argument_name):
    """Parse one launch boolean without silently accepting typos."""
    normalized = str(value).strip().lower()
    if normalized in {'1', 'true', 'yes', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'off'}:
        return False
    raise RuntimeError(
        f'{argument_name} must be one of true/false, 1/0, yes/no, or on/off'
    )


def robot_namespace(index, robot_count, force_namespaced_fleet):
    """Return the namespace while preserving the standalone root contract."""
    if force_namespaced_fleet or robot_count > 1:
        return f'martha_{index}'
    return ''


def launch_gazebo(context):
    """Launch Gazebo with a temporary world at the requested speed factor."""
    source_world = LaunchConfiguration('world').perform(context)
    speed_factor = LaunchConfiguration('sim_speed_factor').perform(context)
    try:
        physics_step_size = float(
            LaunchConfiguration('physics_step_size').perform(context)
        )
    except ValueError as exc:
        raise RuntimeError('physics_step_size must be numeric') from exc
    if physics_step_size < 0.0:
        raise RuntimeError('physics_step_size cannot be negative')
    scaled_world = create_scaled_world(
        source_world,
        speed_factor,
        physics_step_size=(physics_step_size if physics_step_size > 0.0 else None),
    )
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
            # The shared PPO coordinator pauses once all robot pipelines are
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
    force_namespaced_fleet = parse_launch_boolean(
        LaunchConfiguration('force_namespaced_fleet').perform(context),
        'force_namespaced_fleet',
    )
    training_kinematic = parse_launch_boolean(
        LaunchConfiguration('training_kinematic').perform(context),
        'training_kinematic',
    )
    lidar_visualize = parse_launch_boolean(
        LaunchConfiguration('lidar_visualize').perform(context),
        'lidar_visualize',
    )
    try:
        lidar_samples = int(LaunchConfiguration('lidar_samples').perform(context))
    except ValueError as exc:
        raise RuntimeError('lidar_samples must be an integer') from exc
    if lidar_samples < 36 or lidar_samples % 36 != 0:
        raise RuntimeError('lidar_samples must be a positive multiple of 36')

    actions = []
    entity_spawners = []
    controller_spawners = []
    for index in range(robot_count):
        namespace = robot_namespace(
            index,
            robot_count,
            force_namespaced_fleet,
        )
        namespaced_fleet = bool(namespace)
        frame_prefix = f'{namespace}/' if namespace else ''
        entity_name = namespace or 'robot'
        start_x, start_y, _ = (
            parking_pose(index, robot_count)
            if namespaced_fleet
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
                ' training_kinematic:=',
                'true' if training_kinematic else 'false',
                ' lidar_samples:=',
                str(lidar_samples),
                ' lidar_visualize:=',
                'true' if lidar_visualize else 'false',
            ]),
            value_type=str,
        )
        description_topic = robot_description_topic(namespace)
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
        spawn_arguments = [
            '-topic',
            description_topic,
            '-entity',
            entity_name,
        ]
        spawn_arguments.extend([
            '-x',
            str(start_x),
            '-y',
            str(start_y),
            '-z',
            '0.5',
        ])
        spawner = Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            # Keep the factory client itself at the ROS root and let the URDF's
            # node-targeted remaps namespace the plugins without global args.
            name=f'urdf_spawner_{entity_name}',
            arguments=spawn_arguments,
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
        actions.extend([controller_cleanup, state_publisher, ekf])
        if not training_kinematic:
            actions.append(adapter)
        entity_spawners.append(spawner)
        if not training_kinematic:
            controller_spawners.append(controller_spawner)

    # Gazebo's factory remains serialized: each entity must finish before the
    # next one enters spawn_entity. Controller loading is serialized too;
    # controller_manager instances share pluginlib inside gzserver and racing
    # identical class loads can make MultiLibraryClassLoader lose factories.
    # Gazebo itself starts unpaused, so controller switches can still advance.
    for index, spawner in enumerate(entity_spawners[:-1]):
        actions.append(
            RegisterEventHandler(
                OnProcessExit(
                    target_action=spawner,
                    on_exit=[entity_spawners[index + 1]],
                )
            )
        )
    if controller_spawners:
        actions.append(
            RegisterEventHandler(
                OnProcessExit(
                    target_action=entity_spawners[-1],
                    on_exit=[controller_spawners[0]],
                )
            )
        )
        for index, controller_spawner in enumerate(controller_spawners[:-1]):
            actions.append(
                RegisterEventHandler(
                    OnProcessExit(
                        target_action=controller_spawner,
                        on_exit=[controller_spawners[index + 1]],
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
    physics_step_argument = DeclareLaunchArgument(
        'physics_step_size',
        default_value='0.0',
        description='Paso ODE opcional; 0 conserva el valor del world',
    )
    training_kinematic_argument = DeclareLaunchArgument(
        'training_kinematic',
        default_value='false',
        description='Backend planar liviano exclusivo para entrenamiento PPO',
    )
    lidar_samples_argument = DeclareLaunchArgument(
        'lidar_samples',
        default_value='360',
        description='Muestras LiDAR; debe ser multiplo de los 36 sectores PPO',
    )
    lidar_visualize_argument = DeclareLaunchArgument(
        'lidar_visualize',
        default_value='true',
        description='Visualiza los rayos LiDAR en gzclient',
    )
    robot_count_argument = DeclareLaunchArgument(
        'robot_count',
        default_value='1',
        description='Cantidad de Marthas; valores mayores que uno usan namespaces',
    )
    force_namespaced_fleet_argument = DeclareLaunchArgument(
        'force_namespaced_fleet',
        default_value='false',
        description=(
            'Fuerza martha_0 incluso con un robot; usado por el backend shared'
        ),
    )

    # El controlador no publica TF (enable_odom_tf=false). El EKF anterior es
    # la unica autoridad dinamica para odom -> base_link.
    return LaunchDescription([
        world_argument,
        gui_argument,
        speed_argument,
        physics_step_argument,
        training_kinematic_argument,
        lidar_samples_argument,
        lidar_visualize_argument,
        robot_count_argument,
        force_namespaced_fleet_argument,
        OpaqueFunction(function=launch_gazebo),
        OpaqueFunction(function=launch_robots),
    ])
