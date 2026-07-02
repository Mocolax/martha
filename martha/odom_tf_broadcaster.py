import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


class OdomTfBroadcaster(Node):
    def __init__(self):
        super().__init__('odom_tf_broadcaster')

        self.declare_parameter(
            'odometry_topic',
            '/mecanum_drive_controller/odometry',
        )
        odometry_topic = self.get_parameter('odometry_topic').value

        self.warned_missing_frames = False
        self.tf_broadcaster = TransformBroadcaster(self)
        self.subscription = self.create_subscription(
            Odometry,
            odometry_topic,
            self.odometry_callback,
            10,
        )

    def odometry_callback(self, msg):
        if not msg.header.frame_id or not msg.child_frame_id:
            if not self.warned_missing_frames:
                self.get_logger().warn(
                    'Ignoring odometry without parent and child frame IDs'
                )
                self.warned_missing_frames = True
            return

        transform = TransformStamped()
        transform.header.stamp = msg.header.stamp
        transform.header.frame_id = msg.header.frame_id
        transform.child_frame_id = msg.child_frame_id
        transform.transform.translation.x = msg.pose.pose.position.x
        transform.transform.translation.y = msg.pose.pose.position.y
        transform.transform.translation.z = msg.pose.pose.position.z
        transform.transform.rotation = msg.pose.pose.orientation

        self.tf_broadcaster.sendTransform(transform)


def main(args=None):
    rclpy.init(args=args)
    node = OdomTfBroadcaster()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
