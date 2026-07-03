from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import FindExecutable


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
        'mundo_1.world',
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
        PathJoinSubstitution([
            FindPackageShare('gazebo_ros'),
            'launch',
            'gazebo.launch.py',
        ]),
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

    odom_tf_broadcaster_node = Node(
        package=package_name,
        executable='odom_tf_broadcaster',
        name='odom_tf_broadcaster',
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
        odom_tf_broadcaster_node,
    ])
