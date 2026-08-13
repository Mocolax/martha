import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch.substitutions import FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def serial_ports_conflict(esp32_port, lidar_port):
    """Return whether both configured serial paths identify one device."""
    esp32_path = os.path.abspath(os.path.expanduser(esp32_port))
    lidar_path = os.path.abspath(os.path.expanduser(lidar_port))
    if esp32_path == lidar_path:
        return True
    if os.path.exists(esp32_path) and os.path.exists(lidar_path):
        return os.path.samefile(esp32_path, lidar_path)
    return False


def validate_serial_ports(context):
    """Reject a shared serial device before starting either hardware node."""
    start_lidar = LaunchConfiguration('start_lidar').perform(context).lower()
    if start_lidar not in {'true', '1', 'yes', 'on'}:
        return []
    esp32_port = LaunchConfiguration('port').perform(context).strip()
    lidar_port = LaunchConfiguration('lidar_port').perform(context).strip()
    if not esp32_port or not lidar_port:
        raise RuntimeError(
            'Los puertos de la ESP32 y el LiDAR no pueden estar vacios'
        )
    if serial_ports_conflict(esp32_port, lidar_port):
        raise RuntimeError(
            'La ESP32 y el RPLIDAR resuelven al mismo dispositivo serial: '
            f'{esp32_port!r} / {lidar_port!r}'
        )
    return []


def generate_launch_description():
    package_name = 'martha'

    port_argument = DeclareLaunchArgument(
        'port',
        default_value='/dev/ttyUSB0',
        description='Puerto serial de la ESP32',
    )
    lidar_port_argument = DeclareLaunchArgument(
        'lidar_port',
        default_value='/dev/rplidar',
        description='Puerto serial estable del RPLIDAR A2M8',
    )
    lidar_frame_argument = DeclareLaunchArgument(
        'lidar_frame',
        default_value='lidar',
        description='Frame del LaserScan del RPLIDAR',
    )
    lidar_scan_mode_argument = DeclareLaunchArgument(
        'lidar_scan_mode',
        default_value='Sensitivity',
        description='Modo de escaneo del RPLIDAR A2M8',
    )
    start_lidar_argument = DeclareLaunchArgument(
        'start_lidar',
        default_value='true',
        description='Inicia rplidar_node; use false para un driver externo',
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
    lidar_config = PathJoinSubstitution([
        FindPackageShare(package_name),
        'config',
        'rplidar_a2m8.yaml',
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

    rplidar = Node(
        package='rplidar_ros',
        executable='rplidar_node',
        name='rplidar_node',
        parameters=[
            lidar_config,
            {
                'serial_port': LaunchConfiguration('lidar_port'),
                'frame_id': LaunchConfiguration('lidar_frame'),
                'scan_mode': LaunchConfiguration('lidar_scan_mode'),
            },
        ],
        condition=IfCondition(LaunchConfiguration('start_lidar')),
        output='screen',
    )

    return LaunchDescription([
        port_argument,
        lidar_port_argument,
        lidar_frame_argument,
        lidar_scan_mode_argument,
        start_lidar_argument,
        OpaqueFunction(function=validate_serial_ports),
        robot_state_publisher,
        serial_bridge,
        ekf_filter,
        rplidar,
    ])
