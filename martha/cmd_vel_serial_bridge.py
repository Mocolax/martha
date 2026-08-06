import math
import threading
import time

import rclpy
import serial
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Bool
from std_srvs.srv import Trigger


WHEEL_JOINT_NAMES = [
    'base_front_left_wheel_joint',
    'base_front_right_wheel_joint',
    'base_rear_left_wheel_joint',
    'base_rear_right_wheel_joint',
]


def quaternion_from_rpy(roll, pitch, yaw):
    half_roll = roll * 0.5
    half_pitch = pitch * 0.5
    half_yaw = yaw * 0.5

    cr = math.cos(half_roll)
    sr = math.sin(half_roll)
    cp = math.cos(half_pitch)
    sp = math.sin(half_pitch)
    cy = math.cos(half_yaw)
    sy = math.sin(half_yaw)

    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def covariance_6d(diagonal):
    covariance = [0.0] * 36
    for index, value in enumerate(diagonal):
        covariance[index * 6 + index] = value
    return covariance


def covariance_3d(x, y, z):
    return [x, 0.0, 0.0, 0.0, y, 0.0, 0.0, 0.0, z]


class CmdVelSerialBridge(Node):
    def __init__(self):
        super().__init__('cmd_vel_serial_bridge')

        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('baud', 115200)
        self.declare_parameter('input_topic', '/cmd_vel')
        self.declare_parameter('odometry_topic', '/wheel/odometry')
        self.declare_parameter('imu_topic', '/imu/data')
        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('diagnostic_topic', '/diagnostics')
        self.declare_parameter('fault_topic', '/hardware/motor_fault')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('imu_frame', 'imu_link')
        self.declare_parameter('log_serial', False)
        self.declare_parameter('max_vx', 0.35)
        self.declare_parameter('max_vy', 0.35)
        self.declare_parameter('max_wz', 0.80)
        self.declare_parameter('serial_timeout', 1.0)
        self.declare_parameter('startup_grace', 3.0)

        port = self.get_parameter('port').value
        baud = self.get_parameter('baud').value
        input_topic = self.get_parameter('input_topic').value
        odometry_topic = self.get_parameter('odometry_topic').value
        imu_topic = self.get_parameter('imu_topic').value
        joint_state_topic = self.get_parameter('joint_state_topic').value
        diagnostic_topic = self.get_parameter('diagnostic_topic').value
        fault_topic = self.get_parameter('fault_topic').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.imu_frame = self.get_parameter('imu_frame').value
        self.log_serial = self.get_parameter('log_serial').value
        self.max_vx = float(self.get_parameter('max_vx').value)
        self.max_vy = float(self.get_parameter('max_vy').value)
        self.max_wz = float(self.get_parameter('max_wz').value)
        self.serial_timeout = float(
            self.get_parameter('serial_timeout').value
        )
        self.startup_grace = float(self.get_parameter('startup_grace').value)
        numeric_parameters = (
            self.max_vx,
            self.max_vy,
            self.max_wz,
            self.serial_timeout,
            self.startup_grace,
        )
        if not all(math.isfinite(value) for value in numeric_parameters):
            raise ValueError('Bridge limits and timeouts must be finite')
        if min(self.max_vx, self.max_vy, self.max_wz) <= 0.0:
            raise ValueError('Velocity limits must be positive')
        if self.serial_timeout <= 0.0 or self.startup_grace < 0.0:
            raise ValueError(
                'Serial timeout must be positive and grace non-negative'
            )

        self.serial_lock = threading.Lock()
        self.receive_buffer = bytearray()
        self.last_parse_warning = 0.0
        self.motor_fault = True
        self.fault_reason = 'waiting_for_firmware_status'
        self.battery_voltage = math.nan
        self.serial_open_wall_time = time.monotonic()
        self.last_valid_serial_time = self.serial_open_wall_time
        self.serial_seen = False
        self.reset_pending = False
        self.reset_ack_received = False
        self.serial_port = serial.Serial(
            port,
            baudrate=baud,
            timeout=0.01,
            write_timeout=0.10,
            exclusive=True,
        )

        self.subscription = self.create_subscription(
            Twist,
            input_topic,
            self.cmd_vel_callback,
            10,
        )
        self.odometry_publisher = self.create_publisher(
            Odometry,
            odometry_topic,
            10,
        )
        self.imu_publisher = self.create_publisher(
            Imu,
            imu_topic,
            10,
        )
        self.joint_state_publisher = self.create_publisher(
            JointState,
            joint_state_topic,
            10,
        )
        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.fault_publisher = self.create_publisher(
            Bool,
            fault_topic,
            latched_qos,
        )
        self.diagnostic_publisher = self.create_publisher(
            DiagnosticArray,
            diagnostic_topic,
            10,
        )
        self.reset_protection_service = self.create_service(
            Trigger,
            '~/reset_motor_protection',
            self.reset_motor_protection,
        )
        self.reset_odometry_service = self.create_service(
            Trigger,
            '~/reset_odometry',
            self.reset_odometry,
        )
        self.serial_timer = self.create_timer(0.01, self.drain_serial)
        self.serial_watchdog_timer = self.create_timer(
            0.10,
            self.check_serial_watchdog,
        )
        self.diagnostic_timer = self.create_timer(1.0, self.publish_diagnostics)
        self.publish_fault_state()

        self.get_logger().info(
            f'Using {port} at {baud} baud: {input_topic} -> ESP32, '
            f'ESP32 -> {odometry_topic}, {imu_topic} and {joint_state_topic}'
        )

    def cmd_vel_callback(self, msg):
        values = (
            float(msg.linear.x),
            float(msg.linear.y),
            float(msg.angular.z),
        )
        input_is_finite = all(math.isfinite(value) for value in values)
        if not input_is_finite:
            self.reset_pending = False
            self.reset_ack_received = False
            self.set_motor_fault(True, 'invalid_cmd_vel')
            self.get_logger().error(
                'Rejected non-finite cmd_vel and sent zero velocity',
                throttle_duration_sec=1.0,
            )
            values = (0.0, 0.0, 0.0)

        is_zero = all(abs(value) <= 1e-6 for value in values)
        serial_is_fresh = (
            self.serial_seen
            and time.monotonic() - self.last_valid_serial_time
            <= self.serial_timeout
        )
        if (
            input_is_finite
            and self.reset_ack_received
            and serial_is_fresh
            and is_zero
        ):
            self.reset_ack_received = False
            self.set_motor_fault(False, 'ready')

        if self.motor_fault and not is_zero:
            self.get_logger().warning(
                f'Blocking cmd_vel while hardware fault is active: '
                f'{self.fault_reason}',
                throttle_duration_sec=1.0,
            )
            values = (0.0, 0.0, 0.0)

        limited_values = (
            max(-self.max_vx, min(self.max_vx, values[0])),
            max(-self.max_vy, min(self.max_vy, values[1])),
            max(-self.max_wz, min(self.max_wz, values[2])),
        )
        if limited_values != values:
            self.get_logger().warning(
                'Clamped cmd_vel to the configured hardware safety limits',
                throttle_duration_sec=1.0,
            )
        command = (
            f'cmd_vel,{limited_values[0]:.6f},{limited_values[1]:.6f},'
            f'{limited_values[2]:.6f}\n'
        )
        self.write_serial_bytes(command.encode('ascii'))

    def write_serial_bytes(self, payload):
        try:
            with self.serial_lock:
                self.serial_port.write(payload)
            return True
        except (OSError, serial.SerialException) as exc:
            self.reset_pending = False
            self.reset_ack_received = False
            self.set_motor_fault(True, 'serial_write_error')
            self.get_logger().error(
                f'Serial write failed: {exc}',
                throttle_duration_sec=1.0,
            )
            return False

    def mark_serial_alive(self):
        self.last_valid_serial_time = time.monotonic()
        self.serial_seen = True

    def check_serial_watchdog(self):
        now = time.monotonic()
        allowed_age = (
            self.serial_timeout
            if self.serial_seen
            else max(self.serial_timeout, self.startup_grace)
        )
        if now - self.last_valid_serial_time > allowed_age:
            self.reset_pending = False
            self.reset_ack_received = False
            self.set_motor_fault(True, 'serial_timeout')

    def drain_serial(self):
        try:
            with self.serial_lock:
                bytes_waiting = self.serial_port.in_waiting
                if bytes_waiting == 0:
                    return
                self.receive_buffer.extend(self.serial_port.read(bytes_waiting))
        except (OSError, serial.SerialException) as exc:
            self.reset_pending = False
            self.reset_ack_received = False
            self.set_motor_fault(True, 'serial_read_error')
            self.get_logger().error(
                f'Serial read failed: {exc}',
                throttle_duration_sec=1.0,
            )
            return

        while b'\n' in self.receive_buffer:
            raw_line, _, remaining = self.receive_buffer.partition(b'\n')
            self.receive_buffer = bytearray(remaining)
            line = raw_line.decode('utf-8', errors='replace').strip()
            if line:
                if self.process_serial_line(line):
                    self.mark_serial_alive()

        if len(self.receive_buffer) > 2048:
            self.receive_buffer.clear()
            self.warn_parse_error('serial receive buffer overflow')

    def process_serial_line(self, line):
        fields = line.split(',')

        if fields[0] == 'odom':
            values = self.parse_values(fields, expected_fields=7)
            if values is not None:
                self.publish_odometry(values)
                return True
            return False

        if fields[0] == 'imu':
            values = self.parse_values(fields, expected_fields=11)
            if values is not None:
                self.publish_imu(values)
                return True
            return False

        if fields[0] == 'joint':
            values = self.parse_values(fields, expected_fields=9)
            if values is not None:
                self.publish_joint_state(values)
                return True
            return False

        if fields[0] == 'battery':
            values = self.parse_values(fields, expected_fields=2)
            if values is not None:
                self.battery_voltage = values[0]
                return True
            return False

        if fields[0] == 'battery_too_low':
            values = self.parse_values(fields, expected_fields=2)
            if values is not None:
                self.battery_voltage = values[0]
            self.reset_pending = False
            self.reset_ack_received = False
            self.set_motor_fault(True, 'battery_too_low')
            return values is not None

        fault_events = {
            'MOTOR_PROTECTION_ACTIVE': 'motor_protection_active',
            'motor_overcurrent': 'motor_overcurrent',
            'motor_overcurrent_reset_blocked': 'motor_overcurrent_reset_blocked',
            'cmd_vel_blocked: motor protection active': 'cmd_vel_blocked',
        }
        if line in fault_events:
            self.reset_pending = False
            self.reset_ack_received = False
            self.set_motor_fault(True, fault_events[line])
            return True

        if fields[0] == 'battery_reset_blocked':
            values = self.parse_values(fields, expected_fields=2)
            if values is not None:
                self.battery_voltage = values[0]
            self.reset_pending = False
            self.reset_ack_received = False
            self.set_motor_fault(True, 'battery_reset_blocked')
            return values is not None

        if line == 'MOTOR_READY':
            self.reset_pending = False
            if self.serial_seen:
                self.reset_ack_received = True
                self.set_motor_fault(True, 'awaiting_zero_after_ready')
            else:
                self.reset_ack_received = False
                self.set_motor_fault(False, 'ready')
            return True

        if line == 'motor_protection_reset':
            self.reset_pending = False
            self.reset_ack_received = True
            self.set_motor_fault(True, 'awaiting_zero_after_reset')
            return True

        if self.log_serial:
            self.get_logger().info(line)
        return False

    def parse_values(self, fields, expected_fields):
        if len(fields) != expected_fields:
            self.warn_parse_error(
                f'invalid {fields[0]} field count: {len(fields)}'
            )
            return None

        try:
            values = [float(value) for value in fields[1:]]
        except ValueError:
            self.warn_parse_error(f'invalid numeric data in {fields[0]} line')
            return None

        if not all(math.isfinite(value) for value in values):
            self.warn_parse_error(f'non-finite data in {fields[0]} line')
            return None

        return values

    def warn_parse_error(self, message):
        now = time.monotonic()
        if now - self.last_parse_warning >= 1.0:
            self.last_parse_warning = now
            self.get_logger().warning(message)

    def publish_odometry(self, values):
        x, y, yaw, velocity_x, velocity_y, angular_velocity = values
        quaternion = quaternion_from_rpy(0.0, 0.0, yaw)

        message = Odometry()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.odom_frame
        message.child_frame_id = self.base_frame

        message.pose.pose.position.x = x
        message.pose.pose.position.y = y
        message.pose.pose.orientation.x = quaternion[0]
        message.pose.pose.orientation.y = quaternion[1]
        message.pose.pose.orientation.z = quaternion[2]
        message.pose.pose.orientation.w = quaternion[3]
        message.pose.covariance = covariance_6d(
            [0.02, 0.02, 1e6, 1e6, 1e6, 0.05]
        )

        message.twist.twist.linear.x = velocity_x
        message.twist.twist.linear.y = velocity_y
        message.twist.twist.angular.z = angular_velocity
        message.twist.covariance = covariance_6d(
            [0.02, 0.05, 1e6, 1e6, 1e6, 0.05]
        )
        self.odometry_publisher.publish(message)

    def publish_imu(self, values):
        (accel_x, accel_y, accel_z,
         gyro_x, gyro_y, gyro_z,
         roll, pitch, yaw, _stationary) = values
        quaternion = quaternion_from_rpy(roll, pitch, yaw)

        message = Imu()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.imu_frame

        message.orientation.x = quaternion[0]
        message.orientation.y = quaternion[1]
        message.orientation.z = quaternion[2]
        message.orientation.w = quaternion[3]
        message.orientation_covariance = covariance_3d(0.02, 0.02, 0.5)

        message.angular_velocity.x = gyro_x
        message.angular_velocity.y = gyro_y
        message.angular_velocity.z = gyro_z
        message.angular_velocity_covariance = covariance_3d(0.01, 0.01, 0.02)

        message.linear_acceleration.x = accel_x
        message.linear_acceleration.y = accel_y
        message.linear_acceleration.z = accel_z
        message.linear_acceleration_covariance = covariance_3d(0.1, 0.1, 0.1)
        self.imu_publisher.publish(message)

    def publish_joint_state(self, values):
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = WHEEL_JOINT_NAMES
        message.position = values[:4]
        message.velocity = values[4:]
        self.joint_state_publisher.publish(message)

    def set_motor_fault(self, active, reason):
        changed = active != self.motor_fault or reason != self.fault_reason
        if not changed:
            return
        self.motor_fault = bool(active)
        self.fault_reason = reason
        self.publish_fault_state()
        self.publish_diagnostics()
        if active:
            self.get_logger().error(
                f'Hardware motor fault={active}: {reason}'
            )
        else:
            self.get_logger().info(
                f'Hardware motor fault={active}: {reason}'
            )

    def publish_fault_state(self):
        message = Bool()
        message.data = self.motor_fault
        self.fault_publisher.publish(message)

    def publish_diagnostics(self):
        status = DiagnosticStatus()
        status.name = 'martha/esp32_motor_protection'
        status.hardware_id = 'esp32'
        status.level = (
            DiagnosticStatus.ERROR if self.motor_fault else DiagnosticStatus.OK
        )
        status.message = self.fault_reason
        battery_value = (
            f'{self.battery_voltage:.2f}'
            if math.isfinite(self.battery_voltage)
            else 'unknown'
        )
        status.values = [
            KeyValue(key='motor_fault', value=str(self.motor_fault).lower()),
            KeyValue(key='battery_voltage_v', value=battery_value),
            KeyValue(
                key='serial_age_s',
                value=f'{time.monotonic() - self.last_valid_serial_time:.3f}',
            ),
            KeyValue(
                key='reset_pending',
                value=str(self.reset_pending).lower(),
            ),
            KeyValue(
                key='awaiting_zero_after_reset',
                value=str(self.reset_ack_received).lower(),
            ),
        ]
        message = DiagnosticArray()
        message.header.stamp = self.get_clock().now().to_msg()
        message.status = [status]
        self.diagnostic_publisher.publish(message)

    def write_serial_command(self, command):
        return self.write_serial_bytes((command + '\n').encode('ascii'))

    def reset_motor_protection(self, _request, response):
        self.reset_pending = True
        self.reset_ack_received = False
        self.set_motor_fault(True, 'reset_requested')
        zero_sent = self.write_serial_command(
            'cmd_vel,0.000000,0.000000,0.000000'
        )
        reset_sent = self.write_serial_command('reset')
        response.success = bool(zero_sent and reset_sent)
        if response.success:
            response.message = (
                'Reset sent. After firmware acknowledgement, publish a fresh '
                'zero cmd_vel to re-arm the bridge.'
            )
        else:
            response.message = 'Could not write the reset request to the ESP32.'
        return response

    def reset_odometry(self, _request, response):
        response.success = self.write_serial_command('odom_reset')
        response.message = (
            'Odometry reset request sent to the ESP32.'
            if response.success
            else 'Could not write the odometry reset request to the ESP32.'
        )
        return response

    def destroy_node(self):
        if self.serial_port.is_open:
            zero_command = b'cmd_vel,0.000000,0.000000,0.000000\n'
            try:
                with self.serial_lock:
                    self.serial_port.write(zero_command)
                    self.serial_port.flush()
                    self.serial_port.close()
            except (OSError, serial.SerialException):
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelSerialBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
