# Example: ros2 launch martha bringup.launch.py mode:=sim mapping:=true

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.actions import IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition, LaunchConfigurationEquals
from launch.conditions import UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def validate_mode(context):
    mode = LaunchConfiguration('mode').perform(context)
    if mode not in {'sim', 'hardware'}:
        raise RuntimeError(
            "mode debe ser 'sim' o 'hardware'; valor recibido: " + repr(mode)
        )
    return []


def rviz_node(package_name, config_name, use_sim_time):
    return Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=[
            '-d',
            PathJoinSubstitution([
                FindPackageShare(package_name),
                'rviz',
                config_name,
            ]),
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(LaunchConfiguration('rviz')),
        output='screen',
    )


def slam_launch(package_name, config_name, use_sim_time):
    use_sim_time_argument = 'true' if use_sim_time else 'false'
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('slam_toolbox'),
                'launch',
                'online_async_launch.py',
            ])
        ),
        launch_arguments={
            'slam_params_file': PathJoinSubstitution([
                FindPackageShare(package_name),
                'config',
                config_name,
            ]),
            'use_sim_time': use_sim_time_argument,
        }.items(),
    )


def backend_group(package_name, mode, backend_launch, backend_arguments,
                  slam_config, use_sim_time):
    mapping_group = GroupAction(
        condition=IfCondition(LaunchConfiguration('mapping')),
        actions=[
            slam_launch(package_name, slam_config, use_sim_time),
            rviz_node(package_name, 'map.rviz', use_sim_time),
        ],
    )
    odometry_group = GroupAction(
        condition=UnlessCondition(LaunchConfiguration('mapping')),
        actions=[
            rviz_node(package_name, 'lidar.rviz', use_sim_time),
        ],
    )

    return GroupAction(
        condition=LaunchConfigurationEquals('mode', mode),
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        FindPackageShare(package_name),
                        'launch',
                        backend_launch,
                    ])
                ),
                launch_arguments=backend_arguments.items(),
            ),
            mapping_group,
            odometry_group,
        ],
    )


def generate_launch_description():
    package_name = 'martha'
    default_world = PathJoinSubstitution([
        FindPackageShare(package_name),
        'worlds',
        'room.world',
    ])

    mode_argument = DeclareLaunchArgument(
        'mode',
        default_value='sim',
        description="Backend del robot: 'sim' o 'hardware'",
    )
    world_argument = DeclareLaunchArgument(
        'world',
        default_value=default_world,
        description='World de Gazebo; solo se usa con mode:=sim',
    )
    gui_argument = DeclareLaunchArgument(
        'gui',
        default_value='true',
        description='Interfaz de Gazebo; solo se usa con mode:=sim',
    )
    speed_argument = DeclareLaunchArgument(
        'sim_speed_factor',
        default_value='1.0',
        description='Factor de velocidad de Gazebo; solo se usa en simulacion',
    )
    port_argument = DeclareLaunchArgument(
        'port',
        default_value='/dev/ttyUSB0',
        description='Puerto de la ESP32; solo se usa con mode:=hardware',
    )
    lidar_port_argument = DeclareLaunchArgument(
        'lidar_port',
        default_value='/dev/rplidar',
        description='Puerto del RPLIDAR A2M8; solo se usa en hardware',
    )
    lidar_frame_argument = DeclareLaunchArgument(
        'lidar_frame',
        default_value='lidar',
        description='Frame del LaserScan; solo se usa en hardware',
    )
    lidar_scan_mode_argument = DeclareLaunchArgument(
        'lidar_scan_mode',
        default_value='Sensitivity',
        description='Modo del RPLIDAR A2M8; solo se usa en hardware',
    )
    start_lidar_argument = DeclareLaunchArgument(
        'start_lidar',
        default_value='true',
        description='Inicia el RPLIDAR en hardware',
    )
    mapping_argument = DeclareLaunchArgument(
        'mapping',
        default_value='false',
        description='Inicia SLAM Toolbox y usa RViz con frame map',
    )
    rviz_argument = DeclareLaunchArgument(
        'rviz',
        default_value='true',
        description='Inicia RViz',
    )

    simulation_group = backend_group(
        package_name=package_name,
        mode='sim',
        backend_launch='simulation.launch.py',
        backend_arguments={
            'world': LaunchConfiguration('world'),
            'gui': LaunchConfiguration('gui'),
            'sim_speed_factor': LaunchConfiguration('sim_speed_factor'),
        },
        slam_config='SLAM_toolbox_sim.yaml',
        use_sim_time=True,
    )
    hardware_group = backend_group(
        package_name=package_name,
        mode='hardware',
        backend_launch='hardware.launch.py',
        backend_arguments={
            'port': LaunchConfiguration('port'),
            'lidar_port': LaunchConfiguration('lidar_port'),
            'lidar_frame': LaunchConfiguration('lidar_frame'),
            'lidar_scan_mode': LaunchConfiguration('lidar_scan_mode'),
            'start_lidar': LaunchConfiguration('start_lidar'),
        },
        slam_config='SLAM_toolbox_hardware.yaml',
        use_sim_time=False,
    )

    return LaunchDescription([
        mode_argument,
        world_argument,
        gui_argument,
        speed_argument,
        port_argument,
        lidar_port_argument,
        lidar_frame_argument,
        lidar_scan_mode_argument,
        start_lidar_argument,
        mapping_argument,
        rviz_argument,
        OpaqueFunction(function=validate_mode),
        simulation_group,
        hardware_group,
    ])
