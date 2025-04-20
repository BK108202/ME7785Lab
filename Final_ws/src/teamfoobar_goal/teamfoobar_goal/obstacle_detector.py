#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
import math

class ObstacleDetector(Node):
    def __init__(self):
        super().__init__('obstacle_detector')
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, qos)

        # Publishers
        self.trigger_pub = self.create_publisher(Bool, '/trigger_sign', 10)
        self.cmd_pub     = self.create_publisher(Twist, '/cmd_vel', 10)

        # Parameters & state
        self.distance_threshold    = 0.5      # [m]
        self.front_angle_range     = math.radians(15)
        self.prev_object_detected  = False
        self.current_yaw           = 0.0
        self.triggered             = False    # NEW: whether we've already fired this cycle

    def odom_callback(self, msg: Odometry):
        """Track current yaw from odometry for logging or future use."""
        q = msg.pose.pose.orientation
        self.current_yaw = math.atan2(
            2*(q.w*q.z + q.x*q.y),
            1 - 2*(q.y*q.y + q.z*q.z)
        )

    def scan_callback(self, msg: LaserScan):
        """Detect obstacle, align only if needed, then publish trigger_sign once."""
        # Compute indices for ±front_angle_range
        start_idx = int((-self.front_angle_range - msg.angle_min) / msg.angle_increment)
        end_idx   = int(( self.front_angle_range - msg.angle_min) / msg.angle_increment)
        start_idx = max(start_idx, 0)
        end_idx   = min(end_idx, len(msg.ranges) - 1)

        # Check for obstacle in front
        object_detected = any(
            (dist < self.distance_threshold)
            for dist in msg.ranges[start_idx:end_idx+1]
            if math.isfinite(dist)
        )

        trigger_msg = Bool()

        if object_detected:

            # On the rising edge, reset cycle
            if not self.prev_object_detected:
                self.triggered = False

            # If we've already triggered once this cycle, do nothing
            if self.triggered:
                self.prev_object_detected = True
                return

            # Read the two edge distances
            d_left  = msg.ranges[end_idx]
            d_right = msg.ranges[start_idx]
            if not (math.isfinite(d_left) and math.isfinite(d_right)):
                self.prev_object_detected = True
                return

            error        = d_left - d_right
            start_thresh = 0.02    # start aligning if |error| > 5 cm
            finish_tol   = 0.005   # done aligning if |error| < 5 mm

            # If alignment needed
            if abs(error) > start_thresh:
                # Rotate toward the wall
                kp    = 10.0
                omega = max(min(kp * -error, 0.5), -0.5)
                twist = Twist()
                twist.linear.x  = 0.0
                twist.angular.z = omega
                self.cmd_pub.publish(twist)
                self.get_logger().info(f"Aligning: error={error:.3f} m → ω={omega:.2f}")
            else:
                # Either no alignment needed or we're finished aligning
                twist = Twist()  # stop movement
                self.cmd_pub.publish(twist)

                self.get_logger().info(f"Aligned (yaw={self.current_yaw:.2f}); publishing trigger.")
                trigger_msg.data = True
                self.trigger_pub.publish(trigger_msg)

                # Mark that we've triggered—skip further alignment until obstacle clears
                self.triggered = True

            # Mark that obstacle is still present
            self.prev_object_detected = True

        else:
            # No obstacle: reset everything
            trigger_msg.data = False
            self.trigger_pub.publish(trigger_msg)
            self.prev_object_detected = False
            self.triggered           = False

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
