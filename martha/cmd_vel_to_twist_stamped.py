import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from rclpy.node import Node


class CmdVelToTwistStamped(Node):
    def __init__(self):
        super().__init__('cmd_vel_to_twist_stamped')

        self.declare_parameter('input_topic', '/cmd_vel')
        self.declare_parameter('output_topic', '/mecanum_drive_controller/reference')
        self.declare_parameter('frame_id', 'base_link')

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.frame_id = self.get_parameter('frame_id').value

        self.publisher = self.create_publisher(TwistStamped, output_topic, 10)
        self.subscription = self.create_subscription(
            Twist,
            input_topic,
            self.cmd_vel_callback,
            10,
        )

    def cmd_vel_callback(self, msg):
        stamped = TwistStamped()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.header.frame_id = self.frame_id
        stamped.twist = msg
        self.publisher.publish(stamped)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelToTwistStamped()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
