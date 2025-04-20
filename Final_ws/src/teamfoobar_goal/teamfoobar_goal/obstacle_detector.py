#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
import math

class ObstacleDetector(Node):
    def __init__(self):
        super().__init__('obstacle_detector')
        # Use a QoS profile for sensor data.
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
        self.distance_threshold = 0.47  # 50 cm
        # Define the front angular range (±30°).
        self.front_angle_range = math.radians(30)

    def scan_callback(self, msg: LaserScan):
        # Calculate indices corresponding to the front region.
        # Assuming 0 rad is at the center (front) of the scanner.
        start_index = int(( -self.front_angle_range - msg.angle_min) / msg.angle_increment)
        end_index = int(( self.front_angle_range - msg.angle_min) / msg.angle_increment)
        
        # Ensure indices are within the available range.
        start_index = max(start_index, 0)
        end_index = min(end_index, len(msg.ranges) - 1)
        
        object_detected = False
        # Check only within the defined front region.
        for distance in msg.ranges[start_index:end_index+1]:
            if math.isinf(distance) or math.isnan(distance):
                continue
            if distance < self.distance_threshold:
                object_detected = True
                break

        trigger_msg = Bool()
        if object_detected:
            self.get_logger().info("Obstacle detected in front within 50 cm. Triggering sign reading.")
            trigger_msg.data = True
        else:
            trigger_msg.data = False

        self.trigger_pub.publish(trigger_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
