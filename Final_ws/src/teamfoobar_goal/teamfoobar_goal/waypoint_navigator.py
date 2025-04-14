#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose2D, Point
from std_msgs.msg import Int32
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
        # Updated LaserScan subscription to use SENSOR_DATA QoS:
        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor)
        
        # Variables for odometry.
        self.Init = True
        self.Init_ang = 0.0
        self.globalAng = 0.0
        # Use geometry_msgs/Point for positions.
        self.Init_pos = Point()    # Initial position (x, y, z)
        self.globalPos = Point()   # Global position computed relative to the initial pose
        
        # Variables for sign recognition and waypoint control.
        self.recognized_sign = None     # e.g., 1=right, 2=left, 3=back
        self.current_waypoint = None    # Will be a Pose2D (position and orientation)
        self.wall_point = None          # Tuple (x, y) for wall detected in global frame
        self.waypoint_offset = 0.5      # 50 cm offset from the wall
        
        self.get_logger().info("Waypoint Navigator started.")
        self.timer = self.create_timer(0.1, self.timer_callback)
        
    def update_Odometry(self, Odom):
        """
        Update the robot's global position and orientation.
        The global position is calculated in a coordinate frame where the initial pose is (0,0).
        """
        position = Odom.pose.pose.position
        q = Odom.pose.pose.orientation
        orientation = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        
        if self.Init:
            self.Init = False
            self.Init_ang = orientation
            self.globalAng = self.Init_ang
            # Save the initial position.
            self.Init_pos = Point()
            self.Init_pos.x = position.x
            self.Init_pos.y = position.y
            self.Init_pos.z = position.z

        # Compute global position relative to the initial position.
        self.globalPos = Point()
        self.globalPos.x = position.x - self.Init_pos.x
        self.globalPos.y = position.y - self.Init_pos.y
        self.globalPos.z = position.z - self.Init_pos.z
        
        # Compute global angle relative to the initial orientation.
        self.globalAng = orientation - self.Init_ang

    def sign_callback(self, msg: Int32):
        """
        Save the recognized sign from another node.
        For example: 1=right, 2=left, 3=back/turn-around.
        """
        self.recognized_sign = msg.data
        self.get_logger().info(f"Received recognized sign: {self.recognized_sign}")

    def scan_callback(self, msg: LaserScan):
        """
        Use LaserScan data to compute the wall's global position at the angle corresponding
        to the recognized sign.
        """
        if self.recognized_sign is None:
            return

        # Determine the target angle in the robot's frame.
        if self.recognized_sign == 1:       # Right wall
            target_angle = -math.pi / 2
        elif self.recognized_sign == 2:     # Left wall
            target_angle = math.pi / 2
        elif self.recognized_sign == 3:     # Back wall
            target_angle = math.pi
        else:
            # For other signs (e.g., stop/goal), no lateral wall adjustment is needed.
            return

        # Convert target_angle into an index into the LaserScan ranges array.
        index = int((target_angle - msg.angle_min) / msg.angle_increment)
        if index < 0 or index >= len(msg.ranges):
            self.get_logger().warn("Target angle is out of laser scan range.")
            return

        distance = msg.ranges[index]
        if math.isinf(distance) or math.isnan(distance):
            self.get_logger().warn("Invalid distance reading at target angle.")
            return

        # Compute the wall's position in the robot's frame.
        wall_x_robot = distance * math.cos(target_angle)
        wall_y_robot = distance * math.sin(target_angle)
        
        # Transform the wall point from the robot frame to the global frame.
        global_x = self.globalPos.x + wall_x_robot * math.cos(self.globalAng) - wall_y_robot * math.sin(self.globalAng)
        global_y = self.globalPos.y + wall_x_robot * math.sin(self.globalAng) + wall_y_robot * math.cos(self.globalAng)
        self.wall_point = (global_x, global_y)
        self.get_logger().info(f"Wall position (global): ({global_x:.2f}, {global_y:.2f})")

    def timer_callback(self):
        """
        If a recognized sign and a wall point exist, compute a waypoint such that the robot 
        will be positioned 50 cm from the wall. Then, use a proportional controller to drive 
        toward that waypoint.
        """
        # Compute the waypoint based on the wall point and desired offset.
        if self.recognized_sign is not None and self.wall_point is not None:
            wall_x, wall_y = self.wall_point
            # Compute vector from wall to robot (in global frame).
            dx = self.globalPos.x - wall_x
            dy = self.globalPos.y - wall_y
            d = math.hypot(dx, dy)
            if d == 0:
                self.get_logger().warn("Robot is exactly at the wall point; cannot compute offset.")
                return

            # Normalize the vector and multiply by the desired offset (50 cm).
            offset_x = (dx / d) * self.waypoint_offset
            offset_y = (dy / d) * self.waypoint_offset

            # Compute waypoint: position on the line from the wall to the robot,
            # exactly offset by 50 cm from the wall.
            waypoint = Pose2D()
            waypoint.x = wall_x + offset_x
            waypoint.y = wall_y + offset_y
            waypoint.theta = self.globalAng  # Optionally, adjust orientation as needed.
            self.current_waypoint = waypoint
            self.get_logger().info(f"Computed waypoint: ({waypoint.x:.2f}, {waypoint.y:.2f})")
        
        # If a waypoint is set, drive toward it.
        if self.current_waypoint is not None:
            error_x = self.current_waypoint.x - self.globalPos.x
            error_y = self.current_waypoint.y - self.globalPos.y
            distance_error = math.hypot(error_x, error_y)
            
            # When waypoint is reached, reset sign and wall point to avoid repeated recalculations.
            if distance_error < 0.05:
                self.get_logger().info("Waypoint reached.")
                self.current_waypoint = None
                # Resetting sign and wall point after the waypoint is reached.
                self.recognized_sign = None
                self.wall_point = None
                self.stop_robot()
                return

            desired_angle = math.atan2(error_y, error_x)
            angle_error = desired_angle - self.globalAng
            angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))
            
            # Proportional control gains.
            kp_linear = 1.0
            kp_angular = 2.0

            cmd = Twist()
            cmd.linear.x = min(kp_linear * distance_error, 0.2)
            cmd.angular.z = max(min(kp_angular * angle_error, 1.0), -1.0)
            self.cmd_pub.publish(cmd)
            self.get_logger().info(
                f"Driving: distance_error = {distance_error:.2f}, angle_error = {angle_error:.2f}")
        else:
            self.stop_robot()

    def stop_robot(self):
        """Publish zero velocity to stop the robot."""
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
