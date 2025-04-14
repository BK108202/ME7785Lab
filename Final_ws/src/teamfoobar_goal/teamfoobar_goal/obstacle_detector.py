#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

class ObstacleDetector(Node):
    def __init__(self):
        super().__init__('obstacle_detector')
        # Updated LaserScan subscription to use SENSOR_DATA QoS:
        
        qos_profile_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor)
        self.trigger_pub = self.create_publisher(Bool, '/trigger_sign', 10)
        self.distance_threshold = 0.5  # 50 cm

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

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
