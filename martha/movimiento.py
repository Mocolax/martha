import time
import struct
import rclpy
from rclpy.node import Node
from martha.serialCom import Serialhandler
from martha.odometry import Odometria

from sensor_msgs.msg import Joy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Point, Pose, PoseWithCovariance, Twist, TwistWithCovariance
import math


from tf2_ros import TransformBroadcaster


# Use with: ros2 run joy joy_node

class movimiento(Node):
	def __init__(self):
		super().__init__('Movimiento')
  
		self.subscription = self.create_subscription(Joy,'joy', self.listener_callback, 10)
		self.subscription
  
		self.ser = Serialhandler()
  
		self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
		self.odom_proc = Odometria()
  
		self.tf_broadcaster = TransformBroadcaster(self)


	def listener_callback(self, msg_ctrl):
		self.get_logger().info(f"Vx = {msg_ctrl.axes[1]}, Vy = {msg_ctrl.axes[0]}, Wz = {msg_ctrl.axes[2]}")
  
		self.ser.serialSend(msg_ctrl.axes[1], msg_ctrl.axes[0], msg_ctrl.axes[2])
		x, y, th, vx, vy, wz = self.ser.serialRead()

		msg_odom, t = self.odom_proc.Process_Odometry(x, y, th, vx, vy, wz)
		msg_odom.header.stamp = self.get_clock().now().to_msg()
		t.header.stamp = self.get_clock().now().to_msg()
  
		self.odom_pub.publish(msg_odom)
		self.tf_broadcaster.sendTransform(t)

  
   
def main(args=None):
	rclpy.init(args=args)
	nodo = movimiento()
	try:
		rclpy.spin(nodo)
	except KeyboardInterrupt:
		pass
	finally:
		nodo.ser.serialClose()
		nodo.destroy_node()
		rclpy.shutdown()
   
if __name__ == '__main__':
    main()
