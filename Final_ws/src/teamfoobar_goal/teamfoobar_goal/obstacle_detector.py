#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
import math

class ObstacleDetector(Node):
    def __init__(self):
        super().__init__('obstacle_detector')
        qos_profile_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor
        )
        self.trigger_pub = self.create_publisher(Bool, '/trigger_sign', 10)
        self.cmd_pub     = self.create_publisher(Twist, '/cmd_vel', 10)

        self.distance_threshold    = 0.5  # 50 cm
        self.front_angle_range     = math.radians(30)
        self.prev_object_detected  = False
        self.aligned               = False

    def scan_callback(self, msg: LaserScan):
        # compute indices for ±30°
        start_idx = int((-self.front_angle_range - msg.angle_min) / msg.angle_increment)
        end_idx   = int(( self.front_angle_range - msg.angle_min) / msg.angle_increment)
        start_idx = max(start_idx, 0)
        end_idx   = min(end_idx, len(msg.ranges) - 1)

        # detect obstacle in front
        object_detected = False
        for dist in msg.ranges[start_idx:end_idx+1]:
            if math.isinf(dist) or math.isnan(dist):
                continue
            if dist < self.distance_threshold:
                object_detected = True
                break

        trigger_msg = Bool()

        if object_detected:
            # on rising edge, reset alignment
            if not self.prev_object_detected:
                self.aligned = False

            # always publish a ‘False’ trigger at start of alignment
            trigger_msg.data = False
            self.trigger_pub.publish(trigger_msg)

            # alignment phase
            if not self.aligned:
                # read ±30° distances
                left_idx  = end_idx
                right_idx = start_idx
                d_left  = msg.ranges[left_idx]
                d_right = msg.ranges[right_idx]

                # if either reading invalid, skip
                if not (math.isfinite(d_left) and math.isfinite(d_right)):
                    # keep prev_object_detected, skip rest
                    self.prev_object_detected = True
                    return

                error = d_left - d_right
                tol   = 0.01
                self.get_logger().info(
                        f"Aligning: d_left={d_left:.2f}, d_right={d_right:.2f}"
                    )
                if abs(error) > tol:
                    # rotate toward alignment
                    kp    = 1.0
                    omega = max(min(kp * error, 0.5), -0.5)
                    twist = Twist()
                    twist.linear.x  = 0.0
                    twist.angular.z = omega
                    self.cmd_pub.publish(twist)
                    self.get_logger().info(
                        f"Aligning: d_left={d_left:.2f}, d_right={d_right:.2f}, ω={omega:.2f}"
                    )
                else:
                    # aligned: stop and send one True trigger
                    self.aligned = True
                    twist = Twist()
                    twist.linear.x  = 0.0
                    twist.angular.z = 0.0
                    self.cmd_pub.publish(twist)
                    self.get_logger().info("Aligned to wall; publishing trigger.")
                    trigger_msg.data = True
                    self.trigger_pub.publish(trigger_msg)

            self.prev_object_detected = True

        else:
            # no obstacle: reset and clear
            trigger_msg.data = False
            self.trigger_pub.publish(trigger_msg)
            self.prev_object_detected = False
            self.aligned = False

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
    