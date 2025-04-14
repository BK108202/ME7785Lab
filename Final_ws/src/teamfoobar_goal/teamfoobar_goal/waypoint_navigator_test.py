#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose2D
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32
import math
from sensor_msgs.msg import LaserScan

class WaypointNavigator(Node):
    def __init__(self):
        super().__init__('waypoint_navigator')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Int32, '/recognized_sign', self.sign_callback, 10)
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)

        self.current_pose = Pose2D()       # Current pose from odometry.
        self.current_waypoint = None       # Type: Pose2D or None.
        self.waypoint_offset = 0.5         # 50 cm offset

        self.get_logger().info("Waypoint Navigator started.")
        self.timer = self.create_timer(0.1, self.timer_callback)
    
    def scan_callback(self, msg: LaserScan):
        if msg.ranges:
            min_distance = min(msg.ranges)
            trigger_msg = Bool()
            if min_distance < self.distance_threshold:
                self.get_logger().info(f"Obstacle detected at {min_distance:.2f} m. Triggering sign reading.")
                trigger_msg.data = True
            else:
                trigger_msg.data = False
            self.trigger_pub.publish(trigger_msg)
            
    def odom_callback(self, msg: Odometry):
        # Update current pose using odometry.
        self.current_pose.x = msg.pose.pose.position.x
        self.current_pose.y = msg.pose.pose.position.y
        # Convert quaternion to yaw.
        q = msg.pose.pose.orientation
        self.current_pose.theta = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
    
    def sign_callback(self, msg: Int32):
        sign = msg.data
        self.get_logger().info(f"Received sign command: {sign}")
        if sign == 1:      # Right sign.
            self.current_waypoint = self.compute_waypoint('right')
        elif sign == 2:    # Left sign.
            self.current_waypoint = self.compute_waypoint('left')
        elif sign == 3:    # Stop or turn-around sign.
            self.current_waypoint = self.compute_waypoint('back')
        elif sign == 4:    # Goal reached.
            self.current_waypoint = None
            self.stop_robot()
            self.get_logger().info("Goal reached; robot stopped.")
        else:
            self.get_logger().warn("Unknown sign received. Resuming default behavior.")
            self.current_waypoint = None  # Or set to move forward by default.
    
    def compute_waypoint(self, direction: str) -> Pose2D:
        """
        Compute a waypoint relative to the current pose.
        For 'right' and 'left', the waypoint is set 50 cm to that side.
        For 'back', the waypoint is set 50 cm behind.
        """
        waypoint = Pose2D()
        if direction == 'right':
            new_angle = self.current_pose.theta - math.pi/2
        elif direction == 'left':
            new_angle = self.current_pose.theta + math.pi/2
        elif direction == 'back':
            new_angle = self.current_pose.theta + math.pi
        else:
            new_angle = self.current_pose.theta

        waypoint.x = self.current_pose.x + self.waypoint_offset * math.cos(new_angle)
        waypoint.y = self.current_pose.y + self.waypoint_offset * math.sin(new_angle)
        waypoint.theta = new_angle
        
        self.get_logger().info(f"New waypoint ({direction}): x = {waypoint.x:.2f}, y = {waypoint.y:.2f}")
        return waypoint

    def timer_callback(self):
        # If a new waypoint is available, drive toward it.
        if self.current_waypoint is not None:
            error_x = self.current_waypoint.x - self.current_pose.x
            error_y = self.current_waypoint.y - self.current_pose.y
            distance_error = math.hypot(error_x, error_y)
            
            if distance_error < 0.05:
                self.get_logger().info("Waypoint reached.")
                self.current_waypoint = None
                self.stop_robot()
                return

            desired_angle = math.atan2(error_y, error_x)
            angle_error = desired_angle - self.current_pose.theta
            angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))
            
            # Proportional controller gains.
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
