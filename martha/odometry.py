from nav_msgs.msg import Odometry
from tf_transformations import quaternion_from_euler
from geometry_msgs.msg import TransformStamped

class Odometria():
	def __init__():
		pass

	def Process_Odometry(self, x, y, th, vx, vy, wz):
		msg = Odometry()
		msg.header.frame_id = 'odom'          # marco de referencia global
		msg.child_frame_id = 'base_link'      # marco del robot
		qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, th)
		msg.pose.pose.position.x = x
		msg.pose.pose.position.y = y
		msg.pose.pose.position.z = 0.0
  
		msg.pose.pose.orientation.x = qx
		msg.pose.pose.orientation.y = qy
		msg.pose.pose.orientation.z = qz
		msg.pose.pose.orientation.w = qw
		msg.pose.covariance = [0.0]*36
  
		msg.twist.twist.linear.x = vx
		msg.twist.twist.linear.y = vy
		msg.twist.twist.angular.z = wz
		msg.twist.covariance = [0.0]*36
  
  
		t = TransformStamped()
		t.header.stamp = self.get_clock().now().to_msg()
		t.header.frame_id = 'odom'
		t.child_frame_id = 'base_link'
		t.transform.translation.x = x
		t.transform.translation.y = y
		t.transform.translation.z = 0.0
		t.transform.rotation.x = qx
		t.transform.rotation.y = qy
		t.transform.rotation.z = qz
		t.transform.rotation.w = qw
  
		return msg, t

	def generate_tf(self, x, y, qx, qy, qz, qw):
		t = TransformStamped()
		t.header.stamp = self.get_clock().now().to_msg()
		t.header.frame_id = 'odom'
		t.child_frame_id = 'base_link'
		t.transform.translation.x = x
		t.transform.translation.y = y
		t.transform.translation.z = 0.0
		t.transform.rotation.x = qx
		t.transform.rotation.y = qy
		t.transform.rotation.z = qz
		t.transform.rotation.w = qw
