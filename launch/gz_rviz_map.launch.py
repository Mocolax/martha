from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_name = 'martha'
    robot_name = 'robot'

    gui_arg = DeclareLaunchArgument(
        name='gui',
        default_value='true',
    )

    world_file = PathJoinSubstitution([
        FindPackageShare(package_name),
        'worlds',
        'mundo_3.world',
    ])

    world_arg = DeclareLaunchArgument(
        name='world',
        default_value=world_file,
    )

    xacro_file = PathJoinSubstitution([
        FindPackageShare(package_name),
        'urdf',
        'learning.xacro',
    ])

    robot_description = ParameterValue(
        Command([
            FindExecutable(name='xacro'),
            ' ',
            xacro_file,
        ]),
        value_type=str,
    )

    world_launch = IncludeLaunchDescription(
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

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'robot_description': robot_description},
            {'use_sim_time': True},
        ],
        output='screen',
    )

    urdf_spawner_node = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='urdf_spawner',
        arguments=[
            '-topic',
            '/robot_description',
            '-entity',
            robot_name,
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

    mecanum_drive_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['mecanum_drive_controller'],
        output='screen',
    )

    controller_spawners = RegisterEventHandler(
        OnProcessExit(
            target_action=urdf_spawner_node,
            on_exit=[
                joint_state_broadcaster_spawner,
                mecanum_drive_controller_spawner,
            ],
        )
    )

    cmd_vel_adapter_node = Node(
        package=package_name,
        executable='cmd_vel_to_twist_stamped',
        name='cmd_vel_to_twist_stamped',
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    ekf_params_file = PathJoinSubstitution([
        FindPackageShare(package_name),
        'config',
        'ekf.yaml',
    ])

    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        parameters=[
            ekf_params_file,
            {'use_sim_time': True},
        ],
        output='screen',
    )

    slam_params_file = PathJoinSubstitution([
        FindPackageShare(package_name),
        'config',
        'SLAM_toolbox.yaml',
    ])

    slam_toolbox_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('slam_toolbox'),
                'launch',
                'online_async_launch.py',
            ])
        ),
        launch_arguments={
            'slam_params_file': slam_params_file,
            'use_sim_time': 'true',
        }.items(),
    )

    rviz_config = PathJoinSubstitution([
        FindPackageShare(package_name),
        'rviz',
        'map.rviz',
    ])

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    return LaunchDescription([
        gui_arg,
        world_arg,
        world_launch,
        robot_state_publisher_node,
        urdf_spawner_node,
        controller_spawners,
        cmd_vel_adapter_node,
        ekf_node,
        slam_toolbox_launch,
        rviz_node,
    ])
