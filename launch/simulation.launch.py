from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


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
            'world': LaunchConfiguration('world'),
        }.items(),
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
        gazebo,
        robot_state_publisher,
        robot_spawner,
        controller_spawners,
        cmd_vel_adapter,
        ekf_filter,
    ])
