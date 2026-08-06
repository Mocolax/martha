from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch.substitutions import FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_name = 'martha'

    port_argument = DeclareLaunchArgument(
        'port',
        default_value='/dev/ttyUSB0',
        description='Puerto serial de la ESP32',
    )

    xacro_file = PathJoinSubstitution([
        FindPackageShare(package_name),
        'urdf',
        'learning.xacro',
    ])
    bridge_config = PathJoinSubstitution([
        FindPackageShare(package_name),
        'config',
        'hardware_bridge.yaml',
    ])
    ekf_config = PathJoinSubstitution([
        FindPackageShare(package_name),
        'config',
        'ekf_hardware.yaml',
    ])

    robot_description = ParameterValue(
        Command([
            FindExecutable(name='xacro'),
            ' ',
            xacro_file,
        ]),
        value_type=str,
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'robot_description': robot_description},
            {'use_sim_time': False},
        ],
        output='screen',
    )

    serial_bridge = Node(
        package=package_name,
        executable='cmd_vel_serial_bridge',
        name='cmd_vel_serial_bridge',
        parameters=[
            bridge_config,
            {'port': LaunchConfiguration('port')},
        ],
        output='screen',
    )

    ekf_filter = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        parameters=[ekf_config],
        output='screen',
    )

    return LaunchDescription([
        port_argument,
        robot_state_publisher,
        serial_bridge,
        ekf_filter,
    ])
