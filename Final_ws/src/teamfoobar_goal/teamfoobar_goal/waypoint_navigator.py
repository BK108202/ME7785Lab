#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose2D, Point
from std_msgs.msg import Int32, Bool
import math
import numpy as np
from collections import Counter

class WaypointNavigator(Node):
	def __init__(self):
		super().__init__('waypoint_navigator')

		qos_profile_sensor = QoSProfile(
			reliability=ReliabilityPolicy.BEST_EFFORT,
			history=HistoryPolicy.KEEP_LAST,
			depth=10
		)
		
		# Publishers and subscribers.
		self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
		self.create_subscription(Odometry, '/odom', self.update_Odometry, 10)
		self.create_subscription(Int32, '/recognized_sign', self.sign_callback, 10)
		self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor)
		self.create_subscription(Bool, '/trigger_sign', self.trigger_callback, 10)
		
		# --- NEW: once goal reached, stop all processing ---
		self.goal_reached = False
		# Buffer for sign readings
		self.sign_buffer = []

		# Odometry variables.
		self.Init = True
		self.Init_ang = 0.0
		self.globalAng = 0.0
		self.Init_pos = Point()   # initial position (x, y, z)
		self.globalPos = Point()  # global position relative to the initial pose
		
		# Navigation variables.
		self.recognized_sign = None   # from sign recognition (used only to trigger turning)
		self.current_waypoint = None  # computed waypoint (Pose2D)
		self.wall_point = None        # computed wall point (global coordinates)
		self.waypoint_offset = 0.42   # 45 cm offset from the wall (~50 cm as desired)

		# Variables for turning.
		self.turning = False          # Are we currently in a turning phase?
		self.turn_start_angle = None  # Global angle when turn began.
		self.desired_turn_angle = 0.0 # How much to turn (in radians)?

		# Latest obstacle trigger.
		self.trigger = False
		
		self.get_logger().info("Waypoint Navigator started.")
		self.timer = self.create_timer(0.1, self.timer_callback)
		
	def update_Odometry(self, Odom):
		"""Update the robot's global position and orientation relative to the initial pose."""
		position = Odom.pose.pose.position
		q = Odom.pose.pose.orientation
		orientation = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
		
		if self.Init:
			self.Init = False
			self.Init_ang = orientation
			self.globalAng = self.Init_ang
			# Save the initial position.
			self.Init_pos.x = position.x
			self.Init_pos.y = position.y
			self.Init_pos.z = position.z

		self.globalPos.x = position.x - self.Init_pos.x
		self.globalPos.y = position.y - self.Init_pos.y
		self.globalPos.z = position.z - self.Init_pos.z
		self.globalAng = orientation - self.Init_ang

	def sign_callback(self, msg: Int32):
		"""
		Buffer 10 sign readings, pick the most frequent, and then initiate turning or stop.
		"""
		if self.goal_reached:
			return

		# If already turning or navigating, ignore incoming sign until done
		if self.turning or self.current_waypoint is not None:
			self.get_logger().info("Busy processing; ignoring new sign reading.")
			return

		# Collect reading
		self.sign_buffer.append(msg.data)
		self.get_logger().info(f"Sign reading buffered: {msg.data} (size={len(self.sign_buffer)})")
		if len(self.sign_buffer) < 10:
			return

		# Determine majority sign
		vote_counts = Counter(self.sign_buffer)
		common_sign, count = vote_counts.most_common(1)[0]
		self.sign_buffer.clear()
		self.get_logger().info(f"Majority sign: {common_sign} from {count}/10 readings.")

		# Process majority sign
		if common_sign == 5:
			self.get_logger().info("Goal sign detected. Stopping robot.")
			self.stop_robot()
			self.goal_reached = True
			return
		elif common_sign == 1:
			self.desired_turn_angle = math.pi / 2    # Turn left 90°.
		elif common_sign == 2:
			self.desired_turn_angle = -math.pi / 2   # Turn right 90°.
		elif common_sign in (3, 4):
			self.desired_turn_angle = math.pi        # Turn 180°.
		else:
			self.get_logger().warn(f"Unrecognized sign: {common_sign}. Ignoring.")
			return

		# Begin turning phase
		self.turning = True
		self.turn_start_angle = self.globalAng
		self.get_logger().info(
			f"Initiating turn of {self.desired_turn_angle:.2f} radians "
			f"from starting angle {self.turn_start_angle:.2f}."
		)

	def trigger_callback(self, msg: Bool):
		"""Update the obstacle trigger state."""
		if self.goal_reached:
			return

		self.trigger = msg.data
		if self.trigger:
			self.get_logger().info("Obstacle trigger detected.")

	def scan_callback(self, msg: LaserScan):
		"""
		Compute the wall point from the laser scan.
		"""
		if self.goal_reached or self.turning:
			return
		
		target_angle = 0.0
		index = int((target_angle - msg.angle_min) / msg.angle_increment)
		if index < 0 or index >= len(msg.ranges):
			self.get_logger().warn("Front laser scan index out of range.")
			return

		distance = msg.ranges[index]
		if math.isinf(distance) or math.isnan(distance):
			self.get_logger().warn("Invalid distance reading at front.")
			return

		wall_x_robot = distance * math.cos(target_angle)
		wall_y_robot = distance * math.sin(target_angle)
		global_x = (self.globalPos.x +
					wall_x_robot * math.cos(self.globalAng) -
					wall_y_robot * math.sin(self.globalAng))
		global_y = (self.globalPos.y +
					wall_x_robot * math.sin(self.globalAng) +
					wall_y_robot * math.cos(self.globalAng))
		self.wall_point = (global_x, global_y)
		self.get_logger().info(f"Front wall position (global): ({global_x:.2f}, {global_y:.2f})")

	def timer_callback(self):
		"""Main control loop: handle turning, then compute and drive to waypoints."""
		if self.goal_reached:
			return

		# Turning phase
		if self.turning:
			delta = self.globalAng - self.turn_start_angle
			delta = math.atan2(math.sin(delta), math.cos(delta))
			self.get_logger().info(f"Turning: delta = {delta:.2f}, desired = {self.desired_turn_angle:.2f}")
			if abs(delta) >= abs(self.desired_turn_angle) - 0.05:
				self.get_logger().info("Turn complete.")
				self.turning = False
			else:
				kp_turn = 0.5
				error = self.desired_turn_angle - delta
				error = math.atan2(math.sin(error), math.cos(error))
				omega = max(min(kp_turn * error, 0.5), -0.5)
				cmd = Twist()
				cmd.linear.x = 0.0
				cmd.angular.z = omega
				self.cmd_pub.publish(cmd)
			return

		# Wait for wall detection
		if self.wall_point is None:
			cmd = Twist()
			cmd.linear.x = 0.1
			cmd.angular.z = 0.0
			self.cmd_pub.publish(cmd)
			self.get_logger().info("Waiting for wall detection in front...")
			return

		wall_x, wall_y = self.wall_point
		dx = self.globalPos.x - wall_x
		dy = self.globalPos.y - wall_y
		d = math.hypot(dx, dy)
		if d == 0:
			self.get_logger().warn("Robot is exactly at the wall point; cannot compute offset.")
			return
		offset_x = (dx / d) * self.waypoint_offset
		offset_y = (dy / d) * self.waypoint_offset

		final_wp = Pose2D()
		final_wp.x = wall_x + offset_x
		final_wp.y = wall_y + offset_y
		final_wp.theta = self.globalAng

		current_x, current_y = self.globalPos.x, self.globalPos.y
		distance_to_final = math.hypot(final_wp.x - current_x, final_wp.y - current_y)
		max_seg_distance = 0.5

		if distance_to_final > max_seg_distance:
			ratio = max_seg_distance / distance_to_final
			intermediate_wp = Pose2D()
			intermediate_wp.x = current_x + ratio * (final_wp.x - current_x)
			intermediate_wp.y = current_y + ratio * (final_wp.y - current_y)
			intermediate_wp.theta = final_wp.theta
			self.current_waypoint = intermediate_wp
			self.get_logger().info(
				f"Intermediate waypoint set: ({intermediate_wp.x:.2f}, {intermediate_wp.y:.2f})"
			)
		else:
			self.current_waypoint = final_wp
			self.get_logger().info(
				f"Final waypoint set: ({final_wp.x:.2f}, {final_wp.y:.2f})"
			)

		ex, ey = (self.current_waypoint.x - current_x,
					self.current_waypoint.y - current_y)
		dist_err = math.hypot(ex, ey)

		if dist_err < 0.05:
			self.get_logger().info("Waypoint reached.")
			self.current_waypoint = None
			self.stop_robot()
			return

		desired_angle = math.atan2(ey, ex)
		angle_err = desired_angle - self.globalAng
		angle_err = math.atan2(math.sin(angle_err), math.cos(angle_err))

		cmd = Twist()
		cmd.linear.x = min(dist_err, 0.1)
		cmd.angular.z = max(min(2.0 * angle_err, 0.5), -0.5)
		self.cmd_pub.publish(cmd)
		self.get_logger().info(
			f"Driving: distance_error = {dist_err:.2f}, angle_error = {angle_err:.2f}"
		)

	def stop_robot(self):
		"""Stop the robot by publishing zero velocity."""
		cmd = Twist()
		cmd.linear.x = 0.0
		cmd.angular.z = 0.0
		self.cmd_pub.publish(cmd)

def main(args=None):
	rclpy.init(args=args)
	node = WaypointNavigator()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
