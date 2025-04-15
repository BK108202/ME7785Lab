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
        
        # Odometry variables.
        self.Init = True
        self.Init_ang = 0.0
        self.globalAng = 0.0
        self.Init_pos = Point()   # initial position (x, y, z)
        self.globalPos = Point()  # global position relative to the initial pose
        
        # Navigation variables.
        self.recognized_sign = None   # from sign recognition (but used only to trigger turning)
        self.current_waypoint = None  # computed waypoint (Pose2D)
        self.wall_point = None        # computed wall point (global coordinates)
        self.waypoint_offset = 0.45    # 50 cm offset from the wall

        # New variables for turning.
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
        When a sign is received:
          - If sign 5: stop (goal reached).
          - If sign 1, 2, 3, or 4 and we're not already turning or navigating, start turning.
        During turning, ignore further sign messages.
        """
        # Ignore if already in turning mode or if navigating a waypoint.
        if self.turning or self.current_waypoint is not None:
            self.get_logger().info("Already processing a turn/waypoint; ignoring new sign message.")
            return

        # Process the sign.
        if msg.data == 5:
            self.get_logger().info("Goal reached. Stopping robot.")
            self.stop_robot()
            return
        elif msg.data == 1:
            self.desired_turn_angle = math.pi / 2  # Turn left 90 degrees.
        elif msg.data == 2:
            self.desired_turn_angle = -math.pi / 2  # Turn right 90 degrees.
        elif msg.data == 3 or msg.data == 4:
            self.desired_turn_angle = math.pi       # Turn 180 degrees.
        else:
            self.get_logger().warn(f"Unrecognized sign: {msg.data}. Ignoring.")
            return

        self.turning = True
        self.turn_start_angle = self.globalAng
        self.get_logger().info(f"Initiating turn of {self.desired_turn_angle:.2f} radians from starting angle {self.turn_start_angle:.2f}.")
    
    def trigger_callback(self, msg: Bool):
        """Update the obstacle trigger state."""
        self.trigger = msg.data
        if self.trigger:
            self.get_logger().info("Obstacle trigger detected.")

    def scan_callback(self, msg: LaserScan):
        """
        Compute the wall point from the laser scan.
        Now, when not turning, we always use the front reading (target_angle=0).
        """
        if self.turning:
            return  # During turning, ignore laser scan for waypoint planning.
        
        # Use the front: target angle of 0.
        target_angle = 0.0
        index = int((target_angle - msg.angle_min) / msg.angle_increment)
        if index < 0 or index >= len(msg.ranges):
            self.get_logger().warn("Front laser scan index out of range.")
            return

        distance = msg.ranges[index]
        if math.isinf(distance) or math.isnan(distance):
            self.get_logger().warn("Invalid distance reading at front.")
            return

        # Compute wall point in robot's frame.
        wall_x_robot = distance * math.cos(target_angle)
        wall_y_robot = distance * math.sin(target_angle)
        # Transform to global frame.
        global_x = self.globalPos.x + wall_x_robot * math.cos(self.globalAng) - wall_y_robot * math.sin(self.globalAng)
        global_y = self.globalPos.y + wall_x_robot * math.sin(self.globalAng) + wall_y_robot * math.cos(self.globalAng)
        self.wall_point = (global_x, global_y)
        self.get_logger().info(f"Front wall position (global): ({global_x:.2f}, {global_y:.2f})")

    def timer_callback(self):
        """
        Main control loop.
          - If turning is active, command a pure turn until the desired turn is reached.
          - Once turning is complete, use the wall_point (from the front) to compute a waypoint.
          - Then, drive toward the waypoint.
        """
        # --- Turning Phase ---
        if self.turning:
            # Compute how much we have turned so far.
            # Account for wrap-around using atan2 on sine and cosine differences.
            delta = self.globalAng - self.turn_start_angle
            # Normalize the angle difference to [-pi, pi].
            delta = math.atan2(math.sin(delta), math.cos(delta))
            self.get_logger().info(f"Turning: delta = {delta:.2f}, desired = {self.desired_turn_angle:.2f}")
            # Check if the absolute angle turned meets or exceeds the desired turn (with some tolerance).
            if abs(delta) >= abs(self.desired_turn_angle) - 0.05:
                self.get_logger().info("Turn complete.")
                self.turning = False
                # Clear the recognized sign so that subsequent laser scans compute the wall point.
                self.recognized_sign = None
            else:
                # Compute a proportional angular velocity command.
                kp_turn = 1.5  # tuning parameter: adjust as necessary.
                error = self.desired_turn_angle - delta
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = kp_turn * error
                self.cmd_pub.publish(cmd)
                return  # Do not proceed with further navigation until turning is complete.

        # --- Post-turn / Waypoint Phase ---
        # If wall_point is not yet available, wait.
        if self.wall_point is None:
            # Option: Move forward slowly until a valid wall reading is received.
            cmd = Twist()
            cmd.linear.x = 0.1
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            self.get_logger().info("Waiting for wall detection in front...")
            return

        # Compute a waypoint that is offset 50 cm from the wall.
        wall_x, wall_y = self.wall_point
        dx = self.globalPos.x - wall_x
        dy = self.globalPos.y - wall_y
        d = math.hypot(dx, dy)
        if d == 0:
            self.get_logger().warn("Robot is exactly at the wall point; cannot compute offset.")
            return

        offset_x = (dx / d) * self.waypoint_offset
        offset_y = (dy / d) * self.waypoint_offset

        waypoint = Pose2D()
        waypoint.x = wall_x + offset_x
        waypoint.y = wall_y + offset_y
        waypoint.theta = self.globalAng
        self.current_waypoint = waypoint
        self.get_logger().info(f"Computed waypoint: ({waypoint.x:.2f}, {waypoint.y:.2f})")

        # Drive toward the waypoint using proportional control.
        error_x = self.current_waypoint.x - self.globalPos.x
        error_y = self.current_waypoint.y - self.globalPos.y
        distance_error = math.hypot(error_x, error_y)
        
        if distance_error < 0.05:
            self.get_logger().info("Waypoint reached. Ready for next sign command.")
            self.current_waypoint = None
            self.wall_point = None  # Clear wall point so new laser scans update it.
            self.stop_robot()
            return

        desired_angle = math.atan2(error_y, error_x)
        angle_error = desired_angle - self.globalAng
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))

        kp_linear = 1.0
        kp_angular = 2.0

        cmd = Twist()
        cmd.linear.x = min(kp_linear * distance_error, 0.2)
        cmd.angular.z = max(min(kp_angular * angle_error, 1.0), -1.0)
        self.cmd_pub.publish(cmd)
        self.get_logger().info(f"Driving: distance_error = {distance_error:.2f}, angle_error = {angle_error:.2f}")

    def stop_robot(self):
        """Stop the robot by publishing zero velocity."""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)
    
def main(args=None):
    rclpy.init(args=args)
    node = WaypointNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
