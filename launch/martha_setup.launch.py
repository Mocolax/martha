from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node'
        ),
        Node(
            package='martha',
            executable='movimiento',
            name='movimiento'
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': open('src/martha/urdf/martha.urdf').read()}]
        )
    ])
    
    
##
# How to Run: ros2 launch martha martha_setup.launch.py
##
